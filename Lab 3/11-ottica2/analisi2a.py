import numpy as np
import menzalib as mz
import scipy.optimize
import pylab as pl

def f_thetad(h, l): return np.pi/2-np.arctan(h/l)

def lin(x, a, b): return a*x + b

def dsin(x, dx): # Errore su sin(x) al secondo ordine
    return np.abs(np.cos(x)*dx) + np.abs(0.5*np.sin(x)*dx**2)

D, dD = 232.6e-2, 7e-2 # Distanza calibro muro
d, dd = ((9.8+9.8+9.7)/3)*1e-2, 0.1e-2 # distanza tra riflessione e raggio non riflesso

ordine, h1, h2, h3 = np.genfromtxt("dati2a.txt", unpack=True)

h0, dh0 = d/2, dd/2 # Altezza del calibro
h = np.mean(np.array([h1, h2, h3]), axis=0)*1e-2 + h0 # Altezza dei punti sul muro
dh = 0.1e-2 + dh0
thetad = f_thetad(h, D) # Angolo di deflessione e errore
dthetad = np.zeros(len(thetad))
for i in range(len(thetad)): dthetad[i] = np.sqrt(mz.dy(f_thetad, (h[i], D), (dh, dD)))

y = np.sin(thetad)
dy = dsin(thetad, dthetad)
x = np.arange(0, 20, 1)

# Fit e plot
popt, _, dpopt, chi, pval = mz.curve_fitdx(lin, x, y, dy=dy, chi2pval=True)
print("parametri ottimali e errori:\n{}\n{}".format(popt, dpopt))

pl.figure()
pl.subplot(211)
pl.errorbar(x, y, dy, fmt='.')
pl.plot(x, lin(x, *popt))
pl.subplot(212)
pl.plot(x, (y-lin(x, *popt))/dy, '.')
pl.plot(x, x*0)
pl.show()