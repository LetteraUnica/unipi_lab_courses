import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

# R = 72700
# C = 0.1e-6

wt = 305
lenght = 1000
n = 1000
f = 2

fig = plt.figure()
for i in range (0, 10, 1) :
    f *= 2
    w = f*np.pi*2
    T = 2*np.pi/w
    t = np.linspace(-2*T, 2*T, lenght)
    rc = np.zeros(lenght)
    for k in range(1, n, 2) :
        Ak = 1/(np.sqrt(1+(w*k/wt)**2))
        ck = 2/(k*np.pi)
        phik = np.arctan(-w*k/wt)
        rc += Ak * ck * np.sin(w*k*t + phik)
    ax1 = fig.add_subplot(5, 2, i+1)
    ax1.plot(t/T, rc, label="f="+str(f))
    ax1.legend()
    
fig.text(0.5, 0.05, 'Periodo [T]', ha='center', va='center')
fig.text(0.1, 0.5, 'Segnale simulato [arb.un.]', ha='center', va='center', rotation='vertical')
plt.show()