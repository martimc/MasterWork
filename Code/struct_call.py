import os
import ctypes
import pathlib
from math import *
from pdfflow import mkPDFs
from pdfflow import int_me
import tensorflow as tf
import matplotlib.pyplot as plt
from vegasflow import VegasFlow
import numpy.ctypeslib as npct
import numpy as np

pdf = mkPDFs("PDF4LHC15_nlo_mc_pdfas", [0], dirname="/usr/share/lhapdf/PDFsets/")

class struct_test(ctypes.Structure):
    _fields_ = [("m_Z", ctypes.c_double),("m_W", ctypes.c_double),("pi", ctypes.c_double),("G_F", ctypes.c_double),
    ("sin_thetaW", ctypes.c_double),("alpha", ctypes.c_double), ("e", ctypes.c_double),("e_l", ctypes.c_double*2),("thetaf", ctypes.c_double),("m_f", ctypes.c_double),
    ("m_g", ctypes.c_double),("S", ctypes.c_double),("T_f", ctypes.c_double*2),("e_q", ctypes.c_double*2),("L", ctypes.c_double*2),("R", ctypes.c_double*2),
    ("lL", ctypes.c_double*2),("lR", ctypes.c_double*2),("m_sq", ctypes.c_double*2),("S_ij", ctypes.c_double*2*2),("delta", ctypes.c_double*2*2),
    ("sL", ctypes.c_double*2*2*2),("sR", ctypes.c_double*2*2*2),("sLq", ctypes.c_double*2*2*2),("sRq", ctypes.c_double*2*2*2),('M2', ctypes.POINTER(ctypes.c_double)),
    ("lepton_type", ctypes.c_int),("a", ctypes.c_int),("mu_F", ctypes.c_double),("mu_R", ctypes.c_double),("alpha_s", ctypes.POINTER(ctypes.c_double)),('n', ctypes.c_int),
    ('LO_xsec', ctypes.POINTER(ctypes.c_double)), ('susy_xsec', ctypes.POINTER(ctypes.c_double)), ('pid', ctypes.POINTER(ctypes.c_int))
    ]

def init_struct(mass):
    p = struct_test()

    p.m_Z = ctypes.c_double(91.1876)
    p.m_W = ctypes.c_double(80.403)
    p.pi = ctypes.c_double(pi)
    p.G_F = ctypes.c_double(1.16637e-5)
    p.sin_thetaW = ctypes.c_double(1-p.m_W**2/p.m_Z**2)
    p.alpha = ctypes.c_double(sqrt(2) * p.G_F * p.m_W**2 * p.sin_thetaW / pi)
    p.e = sqrt(4*p.alpha*pi)
    p.thetaf = 0#6.25e-1
    p.m_g = 2e3
    p.S = 13e3**2

    p.e_l[0] = ctypes.c_double(0); p.e_l[1] = ctypes.c_double(-1)
    p.e_q[0] = ctypes.c_double(2.0/3); p.e_q[1] = ctypes.c_double(-1*(1.0/3))
    p.S_ij[0][0] = cos(p.thetaf); p.S_ij[0][1] = sin(p.thetaf)
    p.S_ij[1][0] = -1*sin(p.thetaf); p.S_ij[1][1] = cos(p.thetaf)

    for i in range(2):

        p.T_f[i] = ctypes.c_double((-1)**i * (1.0 / 2))
        p.L[i] = ctypes.c_double(2*p.T_f[i] - 2*p.e_q[i] * p.sin_thetaW)
        p.R[i] = ctypes.c_double(-2*p.e_q[i] * p.sin_thetaW)
        p.lL[i] = ctypes.c_double(2*p.T_f[i] - 2 *p.e_l[i]* p.sin_thetaW)
        p.lR[i] = ctypes.c_double(-2 *p.e_l[i]*p.sin_thetaW)
        p.m_sq[i] = ctypes.c_double(1e3)

        for j in range(2):
            if i == j:
                p.delta[i][j] = ctypes.c_double(1)
            else:
                p.delta[i][j] = ctypes.c_double(0)

            for k in range(2):
                p.sL[i][j][k] = ctypes.c_double(p.lL[i] * p.S_ij[j][0] * p.S_ij[k][0])
                p.sR[i][j][k] = ctypes.c_double(p.lR[i] * p.S_ij[j][1] * p.S_ij[k][1])
                p.sLq[i][j][k] = ctypes.c_double(p.L[i] * p.S_ij[j][0] * p.S_ij[k][0])
                p.sRq[i][j][k] = ctypes.c_double(p.R[i] * p.S_ij[j][1] * p.S_ij[k][1])

    p.a = ctypes.c_int(0); p.lepton_type = ctypes.c_int(1);
    p.m_f = ctypes.c_double(mass); p.mu_F = p.mu_R = p.m_f;

    return p

def LO_integrand(xarr, n_dim=None, weight=None):
    M2_test = xarr[:,0]*xarr[:,1]*struct.S
    n = len(xarr[:,0].numpy())

    struct.n = ctypes.c_int(n)
    struct.M2 = npct.as_ctypes(M2_test.numpy())

    struct_point = ctypes.byref(struct)

    q2 = tf.fill(tf.shape(xarr[:,0]), struct.mu_F**2)
    q2 = tf.cast(q2, tf.float64)
    pid = tf.cast([-5,-4,-3,-2,-1,1,2,3,4,5], dtype=tf.int32)
    pid_rev = tf.reverse(pid, [0])

    pid_array = pid.numpy()
    struct.pid = npct.as_ctypes(pid_array)

    pdfs_a = pdf.xfxQ2(pid, xarr[:,0], q2)
    pdfs_b = pdf.xfxQ2(pid_rev, xarr[:,1], q2)

    LO_xsec = np.zeros(struct.n*10)
    struct.LO_xsec = npct.as_ctypes(LO_xsec)

    c_lib.LO_cross(struct_point)
    tf_LO_xsec = tf.cast(LO_xsec, tf.float64)
    tf1_LO_xsec = tf.reshape(tf_LO_xsec, shape=(n, 10))

    xsec_res = tf.reduce_sum((tf1_LO_xsec*pdfs_a*pdfs_b), axis = 1)/(xarr[:,0]*xarr[:,1])

    return xsec_res

@tf.function
def P_qg(z, T_R):
    ones = tf.ones(z.shape, dtype=tf.float64)
    return T_R / 2 * (tf.math.square(z) + tf.math.square(ones - z))

def test_ZNLO(xarr, n_dim = None, weight = None):
    M2 = xarr[:,0]*xarr[:,1]*xarr[:,2]*struct.S
    M2_Z1 = xarr[:,1]*xarr[:,2]*struct.S
    n = len(xarr[:,0].numpy())

    q2 = tf.fill(tf.shape(xarr[:,0]), struct.mu_F**2)
    q2 = tf.cast(q2, tf.float64)
    pid = tf.cast([-5,-4,-3,-2,-1,1,2,3,4,5], dtype=tf.int32)
    pid_rev = tf.reverse(pid, [0])

    pid_array = pid.numpy()
    struct.pid = npct.as_ctypes(pid_array)

    pdfs_a = pdf.xfxQ2(pid, xarr[:,0], q2)
    pdfs_b = pdf.xfxQ2(pid_rev, xarr[:,1], q2)

    struct.alpha_s = npct.as_ctypes(pdf.alphasQ2(q2).numpy())

    struct.n = ctypes.c_int(n)
    struct.M2 = npct.as_ctypes(M2.numpy())
    struct_pointer = ctypes.byref(struct)


def ZNLO_integrand(xarr, n_dim=None, weight=None):
    #struct = init_struct()

    M2_test = xarr[:,0]*xarr[:,1]*xarr[:,2]*struct.S
    M2_test_Z1 = xarr[:,1]*xarr[:,2]*struct.S
    cond_test = M2_test > 4*struct.m_f**2
    cond_test_Z1 = M2_test_Z1 > 4*struct.m_f**2
    n = len(xarr[:,0].numpy())

    idx = tf.where(cond_test).numpy()
    idxZ1 = tf.where(cond_test_Z1).numpy()

    M2 = tf.boolean_mask(M2_test, cond_test)
    M2_Z1 = tf.boolean_mask(M2_test_Z1, cond_test_Z1)
    xarr1_pdf = tf.boolean_mask(xarr[:,1], cond_test)
    xarr1_Z1 = tf.boolean_mask(xarr[:,1], cond_test_Z1)
    xarr2_pdf = tf.boolean_mask(xarr[:,2], cond_test)
    xarr2_Z1 = tf.boolean_mask(xarr[:,2], cond_test_Z1)
    z_calc = tf.boolean_mask(xarr[:,0], cond_test)

    M2_send = M2.numpy()
    N = np.size(M2_send)
    M2Z1_send = M2_Z1.numpy()
    N_Z1 = np.size(M2Z1_send)

    struct_point = ctypes.byref(struct)

    q2 = tf.fill(tf.shape(xarr1_pdf), struct.mu_F**2)
    q2 = tf.cast(q2, tf.float64)
    q2_Z1 = tf.fill(tf.shape(xarr1_Z1), struct.mu_F**2)
    q2_Z1 = tf.cast(q2_Z1, tf.float64)
    pid = tf.cast([-5,-4,-3,-2,-1,1,2,3,4,5,21], dtype=tf.int32)
    pid_array = pid.numpy()

    pdfs_a = pdf.xfxQ2(pid, xarr1_pdf, q2)
    pdfs_b = pdf.xfxQ2(pid, xarr2_pdf, q2)
    pdfsZ1_a = pdf.xfxQ2(pid[:-1], xarr1_Z1, q2_Z1)
    pdfsZ1_b = pdf.xfxQ2(pid[:-1], xarr2_Z1, q2_Z1)

    alpha_s = pdf.alphasQ2(q2)
    alpha_s_Z1 = pdf.alphasQ2(q2_Z1)

    xsec_list = np.zeros(n)

    C_F = 4.0/3
    T_R = 1.0/2
    z2 = tf.math.square(z_calc)
    ones = tf.ones(M2.shape, dtype=tf.float64)
    ones_Z1 = tf.ones(M2_Z1.shape, dtype=tf.float64)

    qq_term1 = -1 * (ones + z2) / (ones - z_calc) * tf.math.log(z_calc)
    qq_term2 = -2 * (ones + z_calc) * tf.math.log(ones - z_calc)
    qq_term3 = -1 * (ones + z_calc) * tf.math.log(M2 / struct.mu_F**2)

    qg_terms = (ones / 2 - z_calc + z2) * tf.math.log(tf.math.square(ones - z_calc) / z_calc) + ones / 4 + 3 * z_calc / 2 - 7 * z2 / 4 + P_qg(z_calc, T_R) / T_R * tf.math.log(M2 / (struct.mu_F**2))

    xa_xb_terms = np.pi**2/3*ones_Z1 - 4*ones_Z1 + 3/2*tf.math.log(M2_Z1/(struct.mu_F**2))

    for k in range(2):
        struct.a = ctypes.c_int(k)

        struct.n = ctypes.c_int(N)
        struct.M2 = npct.as_ctypes(M2_send)
        LO_xsec = np.zeros(struct.n)
        struct.LO_xsec = npct.as_ctypes(LO_xsec)

        c_lib.LO_cross(struct_point)

        struct.n = ctypes.c_int(N_Z1)
        struct.M2 = npct.as_ctypes(M2Z1_send)
        LO_xsecZ1 = np.zeros(struct.n)
        susy_xsec = np.zeros(struct.n)
        struct.LO_xsec = npct.as_ctypes(LO_xsecZ1)
        struct.susy_xsec = npct.as_ctypes(susy_xsec)

        c_lib.LO_cross(struct_point)
        #c_lib.susy_cross(struct_point)

        for pids in pid_array[:-1]:
            if np.mod(pids,2) == k:
                index_1 = np.where(pid_array == pids)[0]
                index_2 = np.where(pid_array == -1*pids)[0]

                frac = alpha_s/np.pi*LO_xsec*pdfs_a[:,index_1[0]]/xarr1_pdf

                qq_term = pdfs_b[:,index_2[0]]/xarr2_pdf*C_F*(qq_term1+qq_term2+qq_term3)
                qg_term = 2*pdfs_b[:,-1]/xarr2_pdf*T_R*qg_terms

                xsec = frac * (qq_term + qg_term)

                #susy_NLO = alpha_s_Z1/(np.pi)*susy_xsec
                xa_xb_xsec = LO_xsecZ1 * (1 + alpha_s_Z1 / (np.pi) * C_F * xa_xb_terms)# + susy_NLO

                xsec_Z1 = xa_xb_xsec*pdfsZ1_a[:,index_1[0]]/xarr1_Z1*pdfsZ1_b[:,index_2[0]]/xarr2_Z1

                for i in range(len(idx)):
                    xsec_list[idx[i][0]] += xsec.numpy()[i]

                for i in range(len(idxZ1)):
                    xsec_list[idxZ1[i][0]] += xsec_Z1.numpy()[i]

    xsec_res = tf.cast(xsec_list, tf.float64)

    return xsec_res

def integration(function, dims, n_calls, struct):

    vegas_instance = VegasFlow(dims, n_calls)

    vegas_instance.compile(function)

    n_iter = 4
    result, err = vegas_instance.run_integration(n_iter)

    return result, err

if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)

    path_to_lib = pathlib.Path().absolute() / "func_lib.so"
    c_lib = npct.load_library('func_lib', os.path.dirname(__file__))

    c_lib.LO_cross.argtypes = [ ctypes.POINTER(struct_test) ]
    c_lib.LO_cross.restype = None


    data = open('slepton_test.dat', 'r')

    mass = []
    prosp_LO_xsec = []
    prosp_NLO_xsec = []

    next(data)
    for lines in data:
        line = lines.split()
        mass.append(float(line[1]))
        prosp_LO_xsec.append(float(line[5]))
        prosp_NLO_xsec.append(float(line[7]))

    prosp_LO_array = np.array(prosp_LO_xsec)

    LO_data = np.zeros(len(mass))

    for i in range(len(mass)):
        struct = init_struct(mass[i])

        result, err = integration(LO_integrand, 2, int(1e4), struct)
        #print("result of LO_xsec at mass: %.3e is: %.5e" % (mass[i], result*0.38938e-3))
        LO_data[i] = result*0.38938e-3*1e15

        print("precent done: ", (i+1)/len(mass)*100)


    with open('LO_data_test.dat', 'w') as file:
        file.write("masses    LO_data\n")
        for i in range(len(LO_data)):
            file.write(("%.5e" + " " + "%.5e" + '\n') % (mass[i], LO_data[i]))



    #hist_data = LO_data-prosp_LO_array/prosp_LO_array

    #plt.hist(hist_data, bins=20)
    #plt.savefig('LO_hist.png')

    #print("LO result is %.5e with error %.5e" % (result*0.38938e-3, err*0.38938e-3))

    print("-----test-----")

    #print("xa_xb cross section at NLO with mf = 114.8 and xa=xb=0.5 is ", ZNLO_integrand(xarr))

    """NLO_data = np.zeros(len(mass))

    for i in range(len(mass)):
        struct = init_struct(mass[i])

        result, err = integration(ZNLO_integrand, 3, int(1e4), struct)
        print("result of NLO_xsec at mass: %.3e is: %.5e" % (mass[i], result*0.38938e-3))
        NLO_data[i] = result*0.38938e-3*1e15

        print("precent done: ", (i+1)/len(mass)*100)

    with open('NLO_data.dat', 'w') as file:
        file.write("masses    NLO_data\n")
        for i in range(len(NLO_data)):
            file.write(("%.5e" + " " + "%.5e" + '\n') % (mass[i], NLO_data[i]))

    result_znlo, err_znlo = integration(ZNLO_integrand, 3, int(1e4), struct)


    print("NLO result is %.5e with error %.5e" % (result_znlo*0.38938e-3, err_znlo*0.38938e-3))"""
