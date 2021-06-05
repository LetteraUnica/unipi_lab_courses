import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.odr import odrpack


def fit_odr(l, d) :
    return 2*np.pi*((l[0]**2/12+d**2) / (9.79*d))**0.5

def modello(l, d) :
    return 2*np.pi*((l**2/12+d**2) / (9.79*d))**0.5


# Lettura dati
percorso = "/Users/Lorenzo/Desktop/Relazioni fisica/6) Pendolo fisico/misureee.txt"
list_periodi = np.genfromtxt(percorso, unpack = True, skip_header = 7)
x = np.array([47.5, 37.52, 27.54, 17.55, 7.57]) / 100
x_err = np.array([0.15, 0.16, 0.16, 0.17, 0.17]) / 100
list_medie_periodi = np.array([])
list_errori_periodi = np.array([])


# Analisi dati
for T in list_periodi :
    list_medie_periodi = np.append(list_medie_periodi, np.average(T))
    list_errori_periodi = np.append(list_errori_periodi, np.std(T, ddof = 1))

list_medie_periodi = list_medie_periodi / 10
list_errori_periodi = list_errori_periodi / (10*(len(list_periodi[0]))**0.5)
chi2 = ((yi - list_medie_periodi) / list_errori_periodi)**2
print("Il chi2 risulta ",chi2.sum())

# Fit l parametro
ddof = 1
model = odrpack.Model(fit_odr)
data = odrpack.RealData(x, list_medie_periodi, sx=x_err, sy=list_errori_periodi)
odr = odrpack.ODR(data, model, beta0=(1., 1.))
out = odr.run()
popt_linear, pcov_linear = out.beta, out.cov_beta
chi2_linear = out.sum_square

print("Parametri ottimali: {}\n".format(popt_linear))
print("Errori parametri: {}\n".format(np.sqrt(np.diagonal(pcov_linear))))
print("Chi2: {}, aspettato {}\n".format(chi2_linear, len(x)-ddof))


# Parte grafici
yi = modello(1.05, x)
x_t = np.linspace(50, 5, 1000) / 100
y_t = modello(1.05, x_t)

# Parametri misurati
plt.figure()
plt.subplot(211)
plt.errorbar(x, list_medie_periodi, list_errori_periodi, x_err, fmt = 'o')
plt.plot(x_t, y_t)
plt.xlabel("[m]")
plt.ylabel("[s]")
plt.title("T(d)")
plt.subplot(212)
plt.errorbar(x, list_medie_periodi-yi, list_errori_periodi, fmt = 'o')
plt.plot(x, x*0)
plt.show()

# Parametri ottimali
plt.figure()
plt.subplot(211)
plt.errorbar(x, list_medie_periodi, list_errori_periodi, x_err, fmt = 'o')
plt.plot(x_t, modello(popt_linear[0], x_t))
plt.xlabel("[m]")
plt.ylabel("[s]")
plt.title("T(d)")
plt.subplot(212)
plt.errorbar(x, list_medie_periodi-modello(popt_linear[0], x), list_errori_periodi, fmt = 'o')
plt.plot(x, x*0)
plt.show()
