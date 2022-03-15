import numpy as np
import matplotlib.pyplot as plt

def norm_dist(x,std,mean):
    dist = 1/(std*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mean)/std)**2)
    return dist

data = open('slepton_test.dat', 'r')

mass_list = []
prosp_LO_xsec = []
prosp_NLO_xsec = []

next(data)
for lines in data:
    line = lines.split()
    mass_list.append(float(line[1]))
    prosp_LO_xsec.append(float(line[5]))
    prosp_NLO_xsec.append(float(line[7]))

prosp_LO_array = np.array(prosp_LO_xsec)

LO_data = np.zeros(len(mass_list))
hist_data = np.zeros(len(mass_list))

data_LO = open('LO_data.dat', 'r')

next(data_LO)
i = 0
for lines in data_LO:
    mass, result = lines.split()
    LO_data[i] = float(result)
    hist_data[i] = (float(result)-prosp_LO_xsec[i])/prosp_LO_xsec[i]
    #print("difference in result at LO is: %.5e with mass %.5e" % (hist_data[i],float(mass_list[i])))
    i += 1

mean_LO = np.mean(hist_data)
std_LO = np.std(hist_data)
x = np.linspace(-0.15,0.15,1001)



plt.hist(hist_data, bins = 40, density=True)
plt.plot(x, norm_dist(x, std_LO, mean_LO), label='normal fit with $\mu$ = %.3e and $\sigma$ = %.3e' % (mean_LO, std_LO))
plt.title("Rel diff between calculated and prospino LO xsec")
plt.xlabel("$(\sigma_{LO}-\sigma_{prospino})/\sigma_{prospino}$")
plt.legend()
plt.savefig('LO_hist.png')
plt.clf()

NLO_data = np.zeros(len(mass_list))
hist_NLO_data = np.zeros(len(mass_list))

data_NLO = open('NLO_data.dat', 'r')

next(data_NLO)
i = 0
for lines in data_NLO:
    mass, result = lines.split()
    LO_data[i] = float(result)
    hist_NLO_data[i] = (float(result)-prosp_NLO_xsec[i])/prosp_NLO_xsec[i]
    #print("difference in result at NLO is: %.5e with mass %.5e" % (hist_NLO_data[i],float(mass_list[i])))
    i += 1

mean_NLO = np.mean(hist_NLO_data)
std_NLO = np.std(hist_NLO_data)
#x = np.linspace(-0.15,0.15,1001)

plt.hist(hist_NLO_data, bins = 40, density = True)
plt.plot(x, norm_dist(x, std_NLO, mean_NLO), label='normal fit with $\mu$ = %.3e and $\sigma$ = %.3e' % (mean_NLO, std_NLO))
plt.title("Rel diff between calculated and prospino NLO xsec w/o SUSY")
plt.xlabel("$(\sigma_{NLO}-\sigma_{prospino})/\sigma_{prospino}$")
plt.legend()
plt.savefig('NLO_hist.png')
plt.clf()
