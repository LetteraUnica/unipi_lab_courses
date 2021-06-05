import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

ft = 1
wt = ft*np.pi*2
f = 100
w = f*np.pi*2

lenght = 1000
n_dati = 500
n = 300

t = np.linspace(-2*T, 2*T, lenght)
delta = np.linspace(0, 1, n_dati)
App = np.zeros(n_dati)
for i in range (n_dati) :
    rc = np.zeros(lenght)
    for k in range(1, n, 1) :
        Ak = 1/(np.sqrt(1+(w*k/wt)**2))
        phik = np.arctan(-w*k/wt)
        ck = 2/(k*np.pi)
        rc += Ak * ck * np.sin(k*np.pi*delta[i]) * np.cos(w*k*t + phik)
    App[i] = np.max(rc)-np.min(rc)

plt.figure()
plt.plot(delta, App)
plt.show()