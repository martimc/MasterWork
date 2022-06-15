import numpy as np
import matplotlib.pyplot as plt


data = open("scale_res_stau.dat", 'r')

masses = []
LO_ress = []
LO_up = []
LO_down = []
NLO_ress = []
NLO_up = []
NLO_down = []

next(data)
for lines in data:
    line = lines.split()
    masses.append(float(line[0]))
    LO_ress.append(float(line[1]))
    LO_up.append(float(line[2]))
    LO_down.append(float(line[3]))
    NLO_ress.append(float(line[4]))
    NLO_up.append(float(line[5]))
    NLO_down.append(float(line[6]))

test = sorted(zip(masses,LO_ress,LO_up,LO_down,NLO_ress,NLO_up,NLO_down))
mass_sort = [x for x,_,_,_,_,_,_ in test]
LO_sort = [x for _,x,_,_,_,_,_ in test]
LOup_sort = [x for _,_,x,_,_,_,_ in test]
LOdown_sort = [x for _,_,_,x,_,_,_ in test]
NLO_sort = [x for _,_,_,_,x,_,_ in test]
NLOup_sort = [x for _,_,_,_,_,x,_ in test]
NLOdown_sort = [x for _,_,_,_,_,_,x in test]

plt.plot(mass_sort,LO_sort, 'b--', label='LO')
plt.plot(mass_sort, NLO_sort, 'g-', label='total NLO')
plt.fill_between(mass_sort, LOup_sort, LOdown_sort, alpha=0.5, hatch='X', color='b')
plt.fill_between(mass_sort, NLOup_sort, NLOdown_sort, alpha=0.5, color='g')
plt.title(r'$pp\rightarrow\tilde{\tau}_1\tilde{\tau}_1^\star$ with $\theta_\tau = 82^\circ$')
plt.xlabel(r'mass $m_l$ (GeV)')
plt.ylabel(r'$\sigma$ (fb)')
plt.yscale('log')
plt.xlim([140,350])
plt.ylim([6*10**(-1), 6*10**(1)])
plt.legend()
plt.savefig('stau_scale_lowmass.png')
