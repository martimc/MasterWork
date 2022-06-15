import numpy as np
import matplotlib.pyplot as plt

data = open("selectron_NLO.dat", 'r')

mass_el = []
stau_mix_el = []
LO_data_el = []
LO_err_el = []
NLO_data_el = []
PDF_err_el = []
alphas_err_el = []

next(data)
for lines in data:
    line = lines.split()
    mass_el.append(float(line[0]))
    stau_mix_el.append(float(line[1]))
    LO_data_el.append(float(line[2]))
    LO_err_el.append(float(line[3]))
    NLO_data_el.append(float(line[4]))
    PDF_err_el.append(float(line[5]))
    alphas_err_el.append(float(line[6]))

sort_el = sorted(zip(mass_el,LO_data_el,NLO_data_el,PDF_err_el,alphas_err_el))
mass_sort_el = [x for x,_,_,_,_ in sort_el]
LO_sort_el = [x for _,x,_,_,_ in sort_el]
NLO_sort_el = [x for _,_,x,_,_ in sort_el]
PDFerr_sort_el = [x for _,_,_,x,_ in sort_el]
alphaerr_sort_el = [x for _,_,_,_,x in sort_el]

full_err_el = np.zeros(len(mass_el))
for i in range(len(mass_el)):
    full_err_el[i] = np.sqrt(PDFerr_sort_el[i]**2 + alphaerr_sort_el[i]**2)/2

data_er = open("selectronR_NLO.dat", 'r')

mass_er = []
stau_mix_er = []
LO_data_er = []
NLO_data_er = []
PDF_err_er = []
alphas_err_er = []

next(data_er)
for lines in data_er:
    line = lines.split()
    mass_er.append(float(line[0]))
    stau_mix_er.append(float(line[1]))
    LO_data_er.append(float(line[2]))
    NLO_data_er.append(float(line[3]))
    PDF_err_er.append(float(line[4]))
    alphas_err_er.append(float(line[5]))

sort_er = sorted(zip(mass_er,LO_data_er,NLO_data_er,PDF_err_er,alphas_err_er))
mass_sort_er = [x for x,_,_,_,_ in sort_er]
LO_sort_er = [x for _,x,_,_,_ in sort_er]
NLO_sort_er = [x for _,_,x,_,_ in sort_er]
PDFerr_sort_er = [x for _,_,_,x,_ in sort_er]
alphaerr_sort_er = [x for _,_,_,_,x in sort_er]

full_err_er = np.zeros(len(mass_er))
for i in range(len(mass_er)):
    full_err_er[i] = np.sqrt(PDFerr_sort_er[i]**2 + alphaerr_sort_er[i]**2)/2

data_t2 = open("stau2_NLO.dat", 'r')

mass_t2 = []
stau_mix_t2 = []
LO_data_t2 = []
NLO_data_t2 = []
PDF_err_t2 = []
alphas_err_t2 = []

next(data_t2)
for lines in data_t2:
    line = lines.split()
    mass_t2.append(float(line[0]))
    stau_mix_t2.append(float(line[1]))
    LO_data_t2.append(float(line[2]))
    NLO_data_t2.append(float(line[3]))
    PDF_err_t2.append(float(line[4]))
    alphas_err_t2.append(float(line[5]))

sort_t2 = sorted(zip(mass_t2,LO_data_t2,NLO_data_t2,PDF_err_t2,alphas_err_t2))
mass_sort_t2 = [x for x,_,_,_,_ in sort_t2]
LO_sort_t2 = [x for _,x,_,_,_ in sort_t2]
NLO_sort_t2 = [x for _,_,x,_,_ in sort_t2]
PDFerr_sort_t2 = [x for _,_,_,x,_ in sort_t2]
alphaerr_sort_t2 = [x for _,_,_,_,x in sort_t2]

full_err_t2 = np.zeros(len(mass_t2))
for i in range(len(mass_t2)):
    full_err_t2[i] = np.sqrt(PDFerr_sort_t2[i]**2 + alphaerr_sort_t2[i]**2)/2

print(np.arccos(stau_mix_t2[0]))

plt.plot(mass_sort_el[:100], LO_sort_el[:100], 'b--', label=r'LO $\tilde{e}_L$')
plt.plot(mass_sort_el[:100], NLO_sort_el[:100], 'b-', label=r'LO+NLO $\tilde{e}_L$')
plt.plot(mass_sort_er[:100], LO_sort_er[:100], 'g--', label=r'LO $\tilde{e}_R$')
plt.plot(mass_sort_er[:100], NLO_sort_er[:100], 'g-', label=r'LO+NLO $\tilde{e}_R$')
#plt.plot(mass_sort_t2[:50], LO_sort_t2[:50], 'r--', label=r'LO $\tilde{\tau}_2$')
#plt.plot(mass_sort_t2[:50], NLO_sort_t2[:50], 'r-', label=r'LO+NLO $\tilde{\tau}_2$')
plt.title(r'$pp\rightarrow\tilde{e}_i\tilde{e}_i^\star$')
plt.xlabel(r'mass $m_l$ (GeV)')
plt.ylabel(r'$\sigma$ (fb)')
plt.yscale('log')
plt.legend()
plt.savefig('LO_NLO_ress.png')
plt.close()

plt.plot(mass_sort_el[:100], full_err_el[:100], 'b-', label=r'$\delta\sigma$ $\tilde{e}_L$')
plt.plot(mass_sort_er[:100], full_err_er[:100], 'g-', label=r'$\delta\sigma$ $\tilde{e}_R$')
#plt.plot(mass_sort_t2[:50], LO_sort_t2[:50], 'r--', label=r'LO $\tilde{\tau}_2$')
#plt.plot(mass_sort_t2[:50], NLO_sort_t2[:50], 'r-', label=r'LO+NLO $\tilde{\tau}_2$')
plt.title(r'The full PDF + $\alpha_s$ uncertainties')
plt.xlabel(r'mass $m_l$ (GeV)')
plt.ylabel(r'$\delta\sigma$ (fb)')
plt.yscale('log')
plt.legend()
plt.savefig('pdf_alpha.png')
plt.close()

rel_err_el = np.zeros(len(mass_t2))
for i in range(len(mass_t2)):
    rel_err_el[i] = PDFerr_sort_el[i]/abs(alphaerr_sort_t2[i])

alphanp_err_el = np.array(alphaerr_sort_el)

plt.plot(mass_sort_el[:100], PDFerr_sort_el[:100], 'b-', label='PDF uncertainty')
plt.plot(mass_sort_el[:100], abs(alphanp_err_el[:100]), 'k-', label=r'$\alpha_s$ uncertainty')
plt.title(r'The separate PDF and $\alpha_s$ uncertainties for $\tilde{e}_L$ production')
plt.xlabel(r'mass $m_l$ (GeV)')
plt.ylabel(r'$\delta\sigma$ (fb)')
plt.yscale('log')
plt.legend()
plt.savefig('ratio_delta.png')
plt.close()
