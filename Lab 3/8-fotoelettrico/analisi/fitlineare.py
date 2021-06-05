import numpy as np
import scipy.optimize
import scipy.constants
import pylab as pl

def lin(x,a,b):
    k = 0.
    x = x/(np.sqrt(1-k**2))
    return a*x+b

def chi_pval(f, x, y, dy, popt):
    chi = np.sum(((y-f(x,*popt))/dy)**2)
    return chi
    
lam, dlam, V0, dV0 = np.genfromtxt("boh.txt", unpack=True)
c = scipy.constants.c
f, df = c/lam * 1e-3, c*dlam/lam**2 * 1e-3

deff = dV0
for i in range(10):    
    popt, pcov = scipy.optimize.curve_fit(lin, f, V0, sigma=deff, absolute_sigma=True)
    deff = np.sqrt(dV0**2 + (df*popt[0])**2)
print(popt, "\n", np.sqrt(np.diag(pcov)))
# print(chi, pval)

t = np.linspace(5e2, 7e2, 1000)
pl.figure()
pl.subplot(211)
pl.grid(linestyle=":")
pl.xlabel("Frequenza [THz]")
pl.ylabel("$V_0$[V]")
pl.errorbar(f, V0, dV0, df, fmt='.', color="black", label="Dati")
pl.plot(t, lin(t, *popt), color="red",  label="fit")
pl.legend()

pl.subplot(212)
pl.grid(linestyle=":")
pl.xlabel("Frequenza [THz]")
pl.errorbar(f, (V0-lin(f, *popt))/deff,  fmt='.', color="black", label="Residui")
pl.plot(t, t*0, color="r", label="Fit")
pl.legend()
pl.show()