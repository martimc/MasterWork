import os
import ctypes
import pathlib
import sys
from pdfflow import mkPDFs
from pdfflow import int_me
import tensorflow as tf
from vegasflow import VegasFlow
import numpy.ctypeslib as npct
import numpy as np

pdf = mkPDFs("PDF4LHC15_nlo_mc_pdfas", [0], dirname="/usr/share/lhapdf/PDFsets/")

class struct_test(ctypes.Structure):
    _fields_ = [("m_Z", ctypes.c_double),("m_W", ctypes.c_double),("pi", ctypes.c_double),("G_F", ctypes.c_double),
    ("sin_thetaW", ctypes.c_double),("alpha", ctypes.c_double), ("e", ctypes.c_double),("e_l", ctypes.c_double*2),("thetaf", ctypes.c_double),("m_f", ctypes.POINTER(ctypes.c_double)),
    ("m_g", ctypes.c_double),("S", ctypes.c_double),("T_f", ctypes.c_double*2),("e_q", ctypes.c_double*2),("L", ctypes.c_double*2),("R", ctypes.c_double*2),
    ("lL", ctypes.c_double*2),("lR", ctypes.c_double*2),("m_sq", ctypes.c_double*2),("S_ij", ctypes.c_double*2*2),("delta", ctypes.c_double*2*2),
    ("sL", ctypes.c_double*2*2*2),("sR", ctypes.c_double*2*2*2),("sLq", ctypes.c_double*2*2*2),("sRq", ctypes.c_double*2*2*2),('M2', ctypes.POINTER(ctypes.c_double)),('z', ctypes.POINTER(ctypes.c_double)),
    ("lepton_type", ctypes.c_int),("mu_F", ctypes.c_double),("mu_R", ctypes.c_double),("alpha_s", ctypes.POINTER(ctypes.c_double)),('n', ctypes.c_int),
    ('LO_xsec', ctypes.POINTER(ctypes.c_double)), ('susy_xsec', ctypes.POINTER(ctypes.c_double)), ('pid', ctypes.POINTER(ctypes.c_int)), ('Z1_xsec', ctypes.POINTER(ctypes.c_double))
    ]

def init_struct(mass, stau_mix):
    p = struct_test()

    p.m_Z = ctypes.c_double(91.1876)
    p.m_W = ctypes.c_double(80.403)
    p.pi = ctypes.c_double(np.pi)
    p.G_F = ctypes.c_double(1.16637e-5)
    p.sin_thetaW = ctypes.c_double(1-p.m_W**2/p.m_Z**2)
    p.alpha = ctypes.c_double(np.sqrt(2) * p.G_F * p.m_W**2 * p.sin_thetaW / np.pi)
    p.e = np.sqrt(4*p.alpha*np.pi)
    p.thetaf = ctypes.c_double(stau_mix)#6.25e-1
    p.m_g = 2e3
    p.S = 13e3**2

    p.e_l[0] = ctypes.c_double(0); p.e_l[1] = ctypes.c_double(-1)
    p.e_q[0] = ctypes.c_double(2.0/3); p.e_q[1] = ctypes.c_double(-1*(1.0/3))
    p.S_ij[0][0] = np.cos(p.thetaf); p.S_ij[0][1] = np.sin(p.thetaf)
    p.S_ij[1][0] = -1*np.sin(p.thetaf); p.S_ij[1][1] = np.cos(p.thetaf)

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

    p.lepton_type = ctypes.c_int(1);
    p.m_f = npct.as_ctypes(mass); p.mu_F = p.mu_R = ctypes.c_double(mass[0]);

    return p

def LO_integrand(xarr, n_dim=None, weight=None):
    M2_test = xarr[:,0]*xarr[:,1]*struct.S
    n = len(xarr[:,0].numpy())

    struct.n = ctypes.c_int(n)
    struct.M2 = npct.as_ctypes(M2_test.numpy())
    struct.z = npct.as_ctypes(np.ones(n))

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
    vec_LO_xsec = tf.cast(LO_xsec, tf.float64)
    tf_LO_xsec = tf.reshape(vec_LO_xsec, shape=(n, 10))

    xsec_res = tf.reduce_sum((tf_LO_xsec*pdfs_a*pdfs_b), axis = 1)/(xarr[:,0]*xarr[:,1])

    return xsec_res

@tf.function
def P_qg(z, T_R):
    ones = tf.ones(z.shape, dtype=tf.float64)
    return T_R / 2 * (tf.math.square(z) + tf.math.square(ones - z))

def NLO_integrand(xarr, n_dim = None, weight = None):
    #M2 = xarr[:,0]*xarr[:,1]*xarr[:,2]*struct.S
    M2 = xarr[:,1]*xarr[:,2]*struct.S
    z = xarr[:,0]
    n = len(z.numpy())
    struct.M2 = npct.as_ctypes(M2.numpy())
    struct.z = npct.as_ctypes(z.numpy())

    q2 = tf.fill(tf.shape(xarr[:,0]), struct.mu_F**2)
    q2 = tf.cast(q2, tf.float64)
    pid = tf.cast([-5,-4,-3,-2,-1,1,2,3,4,5], dtype=tf.int32)
    pid_g = tf.cast([21], dtype=tf.int32)
    pid_rev = tf.reverse(pid, [0])

    pid_array = pid.numpy()
    struct.pid = npct.as_ctypes(pid_array)

    pdfs_a = pdf.xfxQ2(pid, xarr[:,1], q2)
    pdfs_b = pdf.xfxQ2(pid_rev, xarr[:,2], q2)
    pdfs_g = pdf.xfxQ2(pid_g, xarr[:,2], q2)

    alpha_s = pdf.alphasQ2(q2)
    struct.alpha_s = npct.as_ctypes(alpha_s.numpy())

    struct_pointer = ctypes.byref(struct)

    struct.n = ctypes.c_int(n)
    Z1_xsec = np.zeros(struct.n*10)
    LO_xsec = np.zeros(struct.n*10)
    struct.Z1_xsec = npct.as_ctypes(Z1_xsec)
    struct.LO_xsec = npct.as_ctypes(LO_xsec)

    c_lib.LO_cross(struct_pointer)
    c_lib.Z1_cross(struct_pointer)

    vec_LO_xsec = tf.cast(LO_xsec, tf.float64)
    tf_LO_xsec = tf.reshape(vec_LO_xsec, shape=(n, 10))

    vec_Z1_xsec = tf.cast(Z1_xsec, tf.float64)
    tf_Z1_xsec = tf.reshape(vec_Z1_xsec, shape=(n, 10))

    C_F = 4.0/3
    T_R = 1.0/2
    z2 = tf.math.square(z)
    ones = tf.ones(M2.shape, dtype=tf.float64)

    qq_term1 = -1 * (ones + z2) / (ones - z) * tf.math.log(z)
    qq_term2 = -2 * (ones + z) * tf.math.log(ones - z)
    qq_term3 = -1 * (ones + z) * tf.math.log(M2 / struct.mu_F**2)

    qg_terms = (ones / 2 - z + z2) * tf.math.log(tf.math.square(ones - z) / z) + ones / 4 + 3 * z / 2 - 7 * z2 / 4 + P_qg(z, T_R) / T_R * tf.math.log(M2 / (struct.mu_F**2))

    xsec_Z1 = tf.reduce_sum((tf_Z1_xsec*pdfs_a*pdfs_b), axis = 1)/(xarr[:,1]*xarr[:,2])
    qq_term = tf.reduce_sum((tf_LO_xsec*pdfs_a*pdfs_b), axis = 1)*alpha_s*C_F/(np.pi*xarr[:,1]*xarr[:,2])*(qq_term1+qq_term2+qq_term3)
    qg_term = 2 * tf.reduce_sum((tf_LO_xsec*pdfs_a), axis = 1)*pdfs_g*alpha_s*T_R/(np.pi*xarr[:,1]*xarr[:,2])*qg_terms

    NLO_res = xsec_Z1 + qq_term + qg_term
    return NLO_res

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
    c_lib.Z1_cross.agrtypes = [ ctypes.POINTER(struct_test) ]
    c_lib.LO_cross.restype = None

    # set output prefix
    outpref = sys.argv[0] + ' : '

    # check input arguments:
    if len(sys.argv) != 4:
        sys.stdout.write("%s Wrong number of input arguments.\n" % (outpref))
        sys.stdout.write("%s Usage:\n" % (outpref))
        sys.stdout.write("%s   python struct_call.py <input file> <output file> <slepton codes>\n" % (outpref))
        sys.exit()

    infile = sys.argv[1]
    outfile = sys.argv[2]
    out_sleptons = sys.argv[3]


    data = open(infile, 'r')
    next(data)
    mass = []
    stau_mix = []

    if out_sleptons == '1000011_-1000011':
        try:
            for lines in data:
                line = lines.split()
                mass.append([float(line[1]),float(line[1])])
                stau_mix.append(0)
        except:
            pass
    elif out_sleptons == '1000015_-1000015':
        try:
            for lines in data:
                line = lines.split()
                mass.append([float(line[2]),float(line[2])])
                stau_mix.append(np.arccos(float(line[4])))
        except:
            pass
    elif out_sleptons == '2000015_-2000015':
        try:
            for lines in data:
                line = lines.split()
                mass.append([float(line[3]),float(line[3])])
                stau_mix.append(np.arccos(float(line[4])))
        except:
            pass

    LO_data = np.zeros(len(mass))
    NLO_data = np.zeros(len(mass))

    for i in range(2):
        struct = init_struct(np.asarray(mass[i]), stau_mix[i])

        result_LO, err_LO = integration(LO_integrand, 2, int(1e4), struct)
        print("result of LO_xsec at mass: %.3e is: %.5e" % (mass[i][0], result_LO*0.38938e-3*1e15))
        LO_data[i] = result_LO*0.38938e-3*1e15


        result_NLO, err_NLO = integration(NLO_integrand, 3, int(1e4), struct)
        print("result of NLO_xsec test at mass: %.3e is: %.5e" % (mass[i][0], result_NLO*0.38938e-3*1e15))
        NLO_data[i] = result_NLO*0.38938e-3*1e15

        print("precent done: ", (i+1)/len(mass)*100)

    if mass[0][0] == mass[0][1]:
        with open(outfile, 'w') as file:
            file.write("masses    LO_data    NLO_data\n")
            for i in range(len(mass)):
                file.write(("%.5e" + " " + "%.5e" + " " + "%.5e" + " " + "%.5e"+ '\n') % (mass[i][0], np.cos(stau_mix[i]), LO_data[i], NLO_data[i]))
    else:
        with open(outfile, 'w') as file:
            file.write("mass1   mass2    LO_data    NLO_data\n")
            for i in range(len(mass)):
                file.write(("%.5e" + " " + "%.5e" + " " + "%.5e" + " " + "%.5e"+ '\n') % (mass[i][0], mass[i][1], LO_data[i], NLO_data[i]))
