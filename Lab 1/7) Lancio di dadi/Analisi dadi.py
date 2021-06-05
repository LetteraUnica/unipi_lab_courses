import numpy as np
import matplotlib.pyplot as plt
import math

dati = np.genfromtxt("/Users/Lorenzo/Desktop/Relazioni fisica/7) Lancio di dadi/dadi mercoledi/dadi1000_1_15.txt", unpack = True, skip_header = 5)
n_dati = len(dati[1])
somma_dati = sum(dati[1])
x = np.linspace(1, 7, n_dati)
media = np.ones(n_dati) * somma_dati / n_dati

chi2 = 0
for i in range(n_dati) :
    chi2 += (dati[1][i]-media)**2 / media

print(chi2.sum(), "5+-", 10**0.5)
plt.figure()
plt.bar(dati[0], dati[1], 1)
plt.plot(x, media, color = "red")
plt.show()