import math
import numpy as np
import matplotlib.pyplot as plt

def modello(l, d) :
    return 2*np.pi*((l**2/12+d**2) / (9.79*d))**0.5

percorso = "/Users/Lorenzo/Desktop/Relazioni fisica/Pendolo fisico/misureee.txt"
list_periodi = np.genfromtxt(percorso, unpack = True, skip_header = 7)

x = np.array([47.5, 37.52, 27.54, 17.55, 7.57]) / 100
yi = modello(1.05, x)
x_err = np.array([0.15, 0.16, 0.16, 0.17, 0.17]) / 100
list_medie_periodi = np.array([])
list_errori_periodi = np.array([])

for T in list_periodi :
    list_medie_periodi = np.append(list_medie_periodi, np.average(T))
    list_errori_periodi = np.append(list_errori_periodi, np.std(T, ddof = 1))

list_medie_periodi = list_medie_periodi / 10
list_errori_periodi = list_errori_periodi / (10*(len(list_periodi[0]))**0.5)

x_t = np.linspace(50, 5, 1000) / 100
y_t = modello(1.05, x_t)
y_t_plus_sigma = modello(1.052, x_t)
y_t_min_sigma = modello(1.048, x_t)

plt.figure()
plt.errorbar(x, list_medie_periodi, list_errori_periodi, x_err, fmt = 'o')
plt.plot(x_t, y_t)
plt.plot(x_t, y_t_plus_sigma)
plt.plot(x_t, y_t_min_sigma)
plt.xlabel("[m]")
plt.ylabel("[s]")
plt.title("T(d)")
plt.show()

chi = 0
i = 0

while i < 5 :
    chi = chi + (abs(yi[i] - list_medie_periodi[i]) / list_errori_periodi[i])**2
    i = i + 1
print(chi)
residui = (list_medie_periodi-yi)
plt.figure()
plt.errorbar(x, residui, list_errori_periodi, fmt = 'o')
plt.plot(x, x*0)
plt.show()