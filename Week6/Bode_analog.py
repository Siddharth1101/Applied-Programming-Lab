import scipy.signal as sp
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
plt.figure(5)
Hc=sp.lti([-0.1,-1],[1,10**4])
w,mod,phi=Hc.bode()
plt.subplot(2,1,1)
plt.title('Bode plot of magnitude')
plt.xlabel("\u03C9")
plt.ylabel("|H(j\u03A9|")
plt.semilogx(w,(mod))
plt.grid()
plt.subplot(2,1,2)
plt.title('Bode plot of phase')
plt.xlabel("\u03C9")
plt.ylabel("<H(j\u03C9)")
plt.semilogx(w,phi)
plt.grid()
plt.tight_layout()
plt.show()