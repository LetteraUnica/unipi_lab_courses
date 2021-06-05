import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

ft = 1
wt = ft*np.pi*2
f = 100
w = f*np.pi*2
T = 2*np.pi/w

lenght = 500
n = 5000
t = np.linspace(-2*T, 2*T, lenght)
delta = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.95])
fig = plt.figure()
fig1 = plt.figure()
for i in range (0, 8, 1) :
    rc = np.zeros(lenght)
    quadra = np.zeros(lenght)
    for k in range(1, n, 1) :
        Ak = 1/(np.sqrt(1+(w*k/wt)**2))
        phik = np.arctan(-w*k/wt)
        ck = 2/(k*np.pi)
        onda = ck * np.sin(k*np.pi*delta[i])
        quadra += onda * np.cos(w*k*t)
        rc += onda * Ak * np.cos(w*k*t + phik)
    ax = fig.add_subplot(4, 2, i+1)
    ax.plot(t/T, rc, label="delta="+str(np.round(delta[i], 2)))
    ax.legend()
    ax1 = fig1.add_subplot(4, 2, i+1)
    ax1.plot(t/T, rc+delta[i], label="delta="+str(np.round(delta[i], 2)))
    ax1.plot(t/T, quadra+delta[i])
    ax1.legend()
fig.text(0.5, 0.05, 'Periodo [T]', ha='center', va='center')
fig.text(0.1, 0.5, 'Segnale simulato [arb.un.]', ha='center', va='center', rotation='vertical')
plt.show()