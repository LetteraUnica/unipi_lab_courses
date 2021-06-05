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

dati = np.genfromtxt("C:/Users/Lorenzo/Desktop/Relazioni fisica/Piano Inclinato/Dati theta1.txt", unpack = True, skip_header = 1)

tempi = dati[2:]
list_medie = np.array([])
list_errori = np.array([])
for tempo in tempi :
    list_medie = np.append(list_medie, np.average(tempo))
    list_errori = np.append(list_errori, np.std(tempo))
    
list_medie_2 = list_medie**2
list_errori_2 = list_errori * 2 * list_medie
distanze = dati[0][:4]
distanze_errore = np.ones(4) * 0.1

best_fit_1 = regression_line(distanze, list_medie_2[::3], 4)
x = np.linspace(0, 100, 1000)
y = best_fit_1[0] * x + best_fit_1[1]
plt.figure()
plt.xlabel("l [cm]")
plt.ylabel("t^2 [s^2]")
plt.title("theta 1  sfera 1")
plt.plot(x, y)
plt.errorbar(distanze, list_medie_2[::3], yerr = list_errori_2[::3], xerr = distanze_errore, fmt = 'o')
plt.show()

best_fit_2 = regression_line(distanze, list_medie_2[1::3], 4)
y = best_fit_2[0] * x + best_fit_2[1]
plt.figure()
plt.xlabel("l [cm]")
plt.ylabel("t^2 [s^2]")
plt.title("theta 1  sfera 2")
plt.plot(x, y)
plt.errorbar(distanze, list_medie_2[1::3], yerr = list_errori_2[1::3], xerr = distanze_errore, fmt = 'o')
plt.show()

best_fit_3 = regression_line(distanze, list_medie_2[2::3], 4)
y = best_fit_3[0] * x + best_fit_3[1]
plt.figure()
plt.xlabel("l [cm]")
plt.ylabel("t^2 [s^2]")
plt.title("theta 1  sfera 3")
plt.plot(x, y)
plt.errorbar(distanze, list_medie_2[2::3], yerr = list_errori_2[2::3], xerr = distanze_errore, fmt = 'o')
plt.show()

print(best_fit_1)
print(best_fit_2)
print(best_fit_3)