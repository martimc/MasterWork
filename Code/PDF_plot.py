from pdfflow import mkPDFs
from pdfflow import int_me
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



pdf = mkPDFs("PDF4LHC15_nlo_mc_pdfas", [0], dirname="/usr/local/share/LHAPDF/")

x_arr = tf.cast(np.logspace(-3,0,10001), dtype=tf.float64)
x_np = x_arr.numpy()
pid = tf.cast([-2,-1,1,2,3,4,21], dtype=tf.int32)
q2 = tf.cast([10**3]*10001, dtype=tf.float64)

names = np.array([r'$\bar{u}$', r'$\bar{d}$', r'$d$', r'$u$', r'$s$', r'$c$', r'$g$'])

pdfs = pdf.xfxQ2(pid, x_arr, q2)
pdfs_np = pdfs.numpy()

for i in range(7):
    plt.plot(x_np, pdfs[:,i].numpy(), label=names[i])


plt.title(r'plot of $xf_q(x,Q^2)$ at $Q^2=10^3$GeV for the PDF4LHC15 set')
plt.ylim(0,1)
plt.xscale('log')
plt.xlabel(r'$x$')
plt.ylabel(r'$xf_q(x)$')
plt.legend()
plt.savefig('pdf_LHC15_Q1000.png')
