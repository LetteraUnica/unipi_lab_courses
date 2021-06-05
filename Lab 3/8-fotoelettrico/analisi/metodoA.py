import numpy as np
import menzalib as mz
import scipy.optimize
import scipy.constants
import pylab as pl

def lin(x,a,b):
    return a*x+b

def dlin(x, a, b):
    return a

def int_retta(a, b, y0):
    return (y0-b)/a

V, I, dI, digit_I = np.genfromtxt("datiazzurro.txt", unpack=True)
mask = I>0.00
V, I = V[mask]*1e-3, I[mask]*-1
dI, digit_I = dI[mask], digit_I[mask]
dV, dI = mz.dVdig(V), np.sqrt((I*0.04)**2 + dI**2 + (4*digit_I)**2)

popt, pcov, dpopt, chi, pval = mz.curve_fitdx(lin, V, I, dV, dI, chi2pval=True)
print(popt, "\n", dpopt)

cov = np.zeros((3,3))
cov[0:2, 0:2]= pcov
print(cov)
V0 = int_retta(*popt, dpopt[1])
dV0 = np.sqrt(mz.dy(int_retta, (*popt, dpopt[1]), cov))

print(V0, dV0)

pl.plot(V, I, 'o')
pl.show()