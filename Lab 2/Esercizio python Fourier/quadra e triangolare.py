import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

T = 2*np.pi
lenght = 1000
t = np.linspace(-2*T, 2*T, lenght)
n = np.array([1, 3, 5, 9, 49, 99, 499, 999, 4999, 9999])+1

fig = plt.figure()
for i in range (0, 10, 1) :
    quadra = np.zeros(lenght)
    for k in range(1, n[i], 2) :
        quadra += 2/(k*np.pi) * np.sin(k*t)
    ax1 = fig.add_subplot(5, 2, i+1)
    ax1.plot(t/T, quadra, label="n="+str(n[i]))
    ax1.legend()
fig.text(0.5, 0.05, 'Periodo [T]', ha='center', va='center')
fig.text(0.1, 0.5, 'Segnale simulato [arb.un.]', ha='center', va='center', rotation='vertical')
plt.show()

fig = plt.figure()
for i in range(0, 6, 1) :
    triang = np.zeros(lenght)
    for k in range(1, n[i], 2) :
        triang += 4/((k*np.pi)**2) * np.cos(k*t)
    ax1 = fig.add_subplot(3, 2, i+1)
    ax1.plot(t/T, triang, label="n="+str(n[i]))
    ax1.legend()
fig.text(0.5, 0.05, 'Periodo [T]', ha='center', va='center')
fig.text(0.1, 0.5, 'Segnale simulato [arb.un.]', ha='center', va='center', rotation='vertical')
plt.show()