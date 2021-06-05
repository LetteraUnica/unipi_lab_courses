import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

w = 25*np.pi*2
wta = 50*2*np.pi
wtb = 25000*2*np.pi
T = 2*np.pi/w
lenght = 1000
t = np.linspace(-2*T, 2*T, lenght)
onda_out = np.zeros(lenght)

n = 10000
for k in range(1, n, 2) :
    Aka = 1/(np.sqrt(1+(w*k/wta)**2))
    Akb = 1/(np.sqrt(1+(wtb/w*k)**2))
    phika = np.arctan(-w*k/wta)
    phikb = np.arctan(wtb/w*k)
    ck = 2/(k*np.pi)
    onda_out += Aka*Akb*ck*np.sin(w*k*t + phika + phikb)

    
plt.figure()
plt.plot(t/T, onda_out)
plt.show()