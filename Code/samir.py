import numpy as np
import matplotlib.pyplot as plt

"""partonid = np.linspace(1,5,5)
parton2 = np.linspace(-5,-1,5)
partonid = np.append(partonid,21)
partonid = np.append(parton2, partonid)

print(partonid)


for i in range(len(partonid)):
    x_list = []
    xf_list = []
    file = open("PDF4LHC15_nlo_mc_pdfas_0_%d.dat" % partonid[i], 'r')
    lines = file.readlines()
    for line in lines:
        x, q2, xfx = line.split()
        x_list.append(float(x))
        xf_list.append(float(xfx))
    plt.plot(x_list, xf_list)#, label='partonid: %d' % partonid[i])
    #print(xf_list)

plt.ylim([0,1])
plt.xlim([0,1])


plt.savefig('test.png')

print("ferdig")"""
xsec = []
M = []
file = open("xsec_M.dat", 'r')
lines = file.readlines()
for line in lines:
    cross, Ms = line.split()
    xsec.append(float(cross))
    M.append(float(Ms))

N = len(xsec)

dsec = np.zeros(N-1)
M_array = np.zeros(N-1)
for i in range(N-1):
    dsec[i] = (xsec[i+1]-xsec[i])/(M[i+1]-M[i])*M[i]**3
    M_array[i] = M[i]

plt.plot(M_array, dsec)
plt.savefig('test1.png')

print(4.13486e-12 - 4.26624e-12/(510-503.5)*(503.5**3))
