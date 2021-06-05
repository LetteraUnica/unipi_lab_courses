import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

t, V = np.genfromtxt("Battimenti1.txt", unpack=True)

t *= 10**-6     # Converte il tempo in secondi
fft_V = np.abs(np.fft.rfft(V))  # Trasformata di fourier del segnale V

dt_eff = ((t[len(t)-1] - t[0]) / len(t))    # Intervallo medio di tempo tra 2 campionamenti
f_max = 1 / (2*dt_eff)  # Frequenza massima dello spettro
f = np.linspace(0, f_max, len(fft_V))

f = np.delete(f, 0)
fft_V = np.delete(fft_V, 0)     # Elimina il primo dato che corrisponde a f=0 (media di tutti i valori di V)

plt.figure()
plt.subplot(211)
plt.xlabel("Tempo [s]")
plt.ylabel("V(t) [digit]")
plt.plot(t, V, '.-')

plt.subplot(212)
plt.xlabel("Frequenza [Hz]")
plt.ylabel("V(f) [arb. un.]")
plt.yscale("log")   # Inserisce la scala logaritmica sull'asse y
plt.plot(f, fft_V)
plt.show()
