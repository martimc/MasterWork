import numpy as np
import matplotlib.pyplot as plt


data = open('time_file.dat', 'r')

PDF_nr = []
time = []

next(data)
for lines in data:
    line = lines.split()
    PDF_nr.append(float(line[0]))
    time.append(float(line[1]))

data_cpp = open('time_cpp.dat', 'r')

time_cpp = []

next(data_cpp)
for lines in data_cpp:
    line = lines.split()
    time_cpp.append(float(line[1]))

data_multi = open('time_multipdf.dat', 'r')

time_multi = []

next(data_multi)
for lines in data_multi:
    line = lines.split()
    time_multi.append(float(line[1]))


plt.plot(PDF_nr[:-1], time[:-1], label=r'PDFFlow and VegasFlow in Python')
plt.plot(PDF_nr[:-1], time_cpp, label=r'LHAPDF in C++')
plt.plot(PDF_nr[:-1], time_multi[:-1], label=r'Load 10 pdf sets at a time')
plt.xlabel('# of PDFsets computed')
plt.ylabel(r'seconds ($s$)')
plt.legend()
plt.savefig('time_plot.png')
