import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

data = np.genfromtxt("Data/Filtro RC ampiezza.txt", unpack = True, skip_header=1)
Vout = data[0] # Ampiezze misurate in uscita sull'oscilloscopio al variare di f
dVout = data[1]
f_i = data[2] # Frequenza misurata
df = data[3]
Vi = 6.08
dVi = 0.19
A_i = Vout/Vi
dA_i = (1/Vi)*dVout + (Vout/(Vi**2))*dVi
ft = 105
wt = ft*np.pi*2
lenght = 500

AppRC = np.zeros(lenght)
f = np.logspace(1, 5, lenght)
w = f*np.pi*2
T = 2*np.pi/w

n = 100
for i in range(lenght) :
    t = np.linspace(-2*T[i], 2*T[i], lenght)
    rc = np.zeros(lenght)
    for k in range(1, n, 2) :
        Ak = 1/(np.sqrt(1+(w[i]*k/wt)**2))
        ck = 2/(k*np.pi)
        phik = np.arctan(-w[i]*k/wt)
        rc += Ak * ck * np.sin(w[i]*k*t + phik)
    AppRC[i] = max(rc)-min(rc)

plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.errorbar(f_i, A_i, yerr=dA_i, xerr=df, fmt='.')
plt.plot(f, AppRC)
plt.show()