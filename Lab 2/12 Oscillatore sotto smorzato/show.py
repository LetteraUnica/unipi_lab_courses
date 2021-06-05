import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def funzione (t, A, tau, w, phi, Vbias) :
    return A * np.exp(-t/tau) * np.cos(w*t+phi) + Vbias

t, ddp = np.genfromtxt("long3.txt", unpack=True)

x = np.linspace(0, 100000, 4000)
plt.figure()
plt.plot(t, ddp, 'o')
plt.show()