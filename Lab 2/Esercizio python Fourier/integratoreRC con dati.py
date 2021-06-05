import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

# R = 72700
# C = 0.1e-6

f = 575 # Valori consentiti 25, 50, 100, 200, 400, 575
data = np.genfromtxt("Circuito RC/"+str(f)+"hZ.txt", unpack = True)
w = f*np.pi*2
wt = 305
T = 2*np.pi/w
lenght = 10000
t = np.linspace(0, 0.25, lenght)
rc = np.zeros(lenght)

n = 10000
for k in range(1, n, 2) :
    Ak = 1/(np.sqrt(1+(w*k/wt)**2))
    ck = 2/(k*np.pi)
    phik = np.arctan(-w*k/wt)
    rc += Ak * ck * np.sin(w*k*t + phik)
    
A_dati = np.average(data[1])
if (f<=50) :
    A_dati += 10 # L'offset aggiuntivo  di 10 vale per frequenze basse (25 e 50 Hz)

plt.figure()
plt.title("Integratore RC fourier con dati")
plt.ylim((490,616)) # Da cambiare in base alla frequenza
plt.xlim((0,7)) # Da cambiare in base alla frequenza
plt.ylabel('ddp [digit]')
plt.xlabel('Time [ms]')
plt.errorbar(data[0]/1000, data[1], fmt = 'o', label="Data "+str(f)+"Hz")
plt.plot((t-T/2)*1000, 883*rc+A_dati, label="Simulazione") # Il fattore moltiplicativo 883 è la differenza tra il massimo e il minimo dei dati acquisiti a 25Hz
plt.legend()
plt.show()