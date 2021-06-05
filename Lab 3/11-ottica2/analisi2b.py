import numpy as np
import menzalib as mz
import scipy.optimize
import pylab as pl

# Analisi 2B, interferometro

def lamda_mercurio(nfrange, s, k): return 2*s*k/nfrange
def const(x, a): return a

# Calcolo di k
nfrange = 80
s = 14e-5   
ds = 0.5e-5
lam = 632.8e-9
k, dk = lam*nfrange/(2*s), lam*nfrange/(2*s**2)*ds
print("k = {:.3f} +- {:.3f}".format(k, dk))

# Calcolo della lunghezza d'onda del mercurio
frange_hg, s = np.genfromtxt("dati2b.txt", unpack=True)
lambda_hg = lamda_mercurio(frange_hg, s, k)
dlamda_hg = np.zeros(len(lambda_hg))
for i in range(len(s)):
    dlamda_hg[i] = np.sqrt(mz.dy(lamda_mercurio, (frange_hg[i], s[i], k), (0, ds, dk)))

# Siccome abbiamo preso 5 set di dati faccio una media delle lunghezze d'onda
x = [1, 2, 3, 4, 5]
popt, _, dpopt, chi, pval = mz.curve_fitdx(const, x, lambda_hg, dy=dlamda_hg, chi2pval=True)
print("lambda_hg = {:.8f} +- {:.8f}".format(popt[0], dpopt[0]))

pl.errorbar(x, lambda_hg, dlamda_hg, fmt='o', label="dati")
pl.plot(x, [popt]*5, label="media")
pl.legend()
pl.show()