import math
import numpy as np
import matplotlib.pyplot as plt

def regression_line (datix, datiy, n_dati) : #Calcola la linea di miglior fit
    ux = np.average(datix)                   #Formule prese da http://mathworld.wolfram.com/LeastSquaresFitting.html
    uy = np.average(datiy)
    sxx = -n_dati * ux**2
    sxy = -n_dati * ux * uy
    syy = -n_dati * uy**2
    
    i = 0
    while i < n_dati :
        sxx = sxx + datix[i]**2
        sxy = sxy + datix[i] * datiy[i]
        syy = syy + datiy[i]**2
        i = i + 1
    
    b = sxy / sxx
    a = uy - b * ux
    s = np.sqrt((syy - sxy**2 / sxx) / (n_dati - 2))
    delta_b = s / np.sqrt(sxx)
    delta_a = s * np.sqrt(1 / n_dati + ux**2 / sxx)
    return np.array([b, a, delta_b, delta_a])

dati = np.genfromtxt("C:/Users/Lorenzo/Desktop/Relazioni fisica/Piano Inclinato/Dati theta2.txt", unpack = True, skip_header = 1)

tempi = dati[2:]
list_medie = np.array([])
list_errori = np.array([])
for tempo in tempi :
    list_medie = np.append(list_medie, np.average(tempo))
    list_errori = np.append(list_errori, np.std(tempo))


distanze = dati[0][:4]
distanze_errore = np.ones(4) * 0.1
best_fit = regression_line(np.log10(distanze), np.log10(list_medie[::3]), 4)
x = np.linspace(1.4, 2.6, 1000)
y = x * best_fit[0] + best_fit[1]
print(best_fit)

plt.figure()
plt.xlabel("t [s]")
plt.ylabel("l [cm]")
plt.errorbar(np.log10(distanze), np.log10(list_medie[::3]), yerr = list_errori[::3] / (np.log(10) * list_medie[::3]), xerr = distanze_errore / (np.log(10) * distanze), fmt = 'o')
plt.plot(x, y)
plt.show()