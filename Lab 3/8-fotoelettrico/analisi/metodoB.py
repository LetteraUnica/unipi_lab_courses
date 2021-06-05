import numpy as np
import menzalib as mz
import scipy.optimize
import pylab as pl
import numdifftools.nd_algopy as nda

def step(a, a0):
    a, a0 = np.array(a), np.array(a0)
    return a<a0

def I_V(V, a, b, V0, I0):
    return a*step(V, V0)*(V0-V)**2 + b*V + I0

dI_V = nda.Derivative(I_V)

V, I, dI, digit_I = np.genfromtxt("datiazzurro.txt", unpack=True)
V, I = V*1e-3, I*-1
dV, dI = mz.dVdig(V), np.sqrt((I*0.04)**2 + dI**2 + (4*digit_I)**2)
popt, pcov, dpopt, chi2, pval = mz.curve_fitdx(I_V, V, I, dV, dI, (1,1,0.8,0.1), chi2pval=True)
print(popt, "\n", dpopt)
print(chi2, pval)
#mz.mat_tex(mz.ne_tex([V, I], [dV, dI]), titolo="Tensione frenante[V] & Corrente catodica[nA]")
mz.mat_tex(mz.ns_tex([[chi2, pval]]))


t = np.linspace(0, 1.7, 1000)
deff = np.sqrt(dI**2 + (dV*dI_V(V, *popt))**2)
pl.figure()
pl.subplot(211)
pl.title("Fit giallo $\lambda=577$nm")
pl.grid(linestyle=":")
pl.xlabel("Tensione frenante [V]")
pl.ylabel("Corrente catodica [nA]")
pl.errorbar(V,  I, dI, dV, '.', color="black", label="Dati")
pl.plot(t, I_V(t, *popt), color="r", label="Fit")
pl.legend()

pl.subplot(212)
pl.grid(linestyle=":")
pl.xlabel("Tensione frenante [V]")
pl.plot(V, (I-I_V(V, *popt))/deff, '.', color="black", label="Residui")
pl.plot(t, t*0, color="r", label="Fit")
pl.legend()

pl.show()