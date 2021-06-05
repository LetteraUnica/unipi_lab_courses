import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize, scipy.stats

# Definisco il modello
def modello(d, l) :
    return 2 * np.pi * ((l ** 2 / 12 + d ** 2) / (9.79 * d)) ** 0.5

def modello_2rd(d, l, theta) :
    return 2 * np.pi * ((l ** 2 / 12 + d ** 2) / (9.79 * d)) ** 0.5 * (1 + theta**2/16)

def derivata_modello(d, l) :
    alpha = 12*d + (l**2)/d
    return abs((3.47816*d**2 - 0.28985*l**2) / (d**2 * np.sqrt(alpha)))


# Lettura misure
periodi = np.genfromtxt("C:/Users/Lorenzo/Desktop/Relazioni fisica/6) Pendolo fisico/misureee.txt", unpack=True, skip_header=7)
d = np.array([47.5, 37.52, 27.54, 17.55, 7.57]) / 100
errore_d = np.array([0.15, 0.16, 0.16, 0.17, 0.17]) / 100
medie_periodi = np.array([])
errori_periodi = np.array([])

for T in periodi :
    medie_periodi = np.append(medie_periodi, np.average(T))
    errori_periodi = np.append(errori_periodi, np.std(T, ddof = 1))

medie_periodi = medie_periodi / 10
errori_periodi = errori_periodi / (10*(len(periodi[0]))**0.5)


# Parte analisi dati
# Fit con l parametro
[popt_l, pcov_l] = scipy.optimize.curve_fit(modello, d, medie_periodi, [1], errori_periodi)
chi2 = ((medie_periodi-modello(d, popt_l[0])) / (errori_periodi))**2
pvalue = 1 - scipy.stats.chi2.cdf(np.sum(chi2), 4)
print("FIT l PARAMETRO\n")
print("l ottimale : {}+-{}\n".format(popt_l[0], np.diagonal(pcov_l)[0]))
print("Chi2 singolo: {}\n".format(chi2))
print("Chi2 complessivo: {}\n".format(np.sum(chi2)))
print("Pvalue: {}\n\n".format(pvalue))

# Fit con l e theta parametri
[popt_ltheta, pcov_ltheta] = scipy.optimize.curve_fit(modello_2rd, d, medie_periodi, [1, 1], errori_periodi)
chi2 = ((medie_periodi-modello_2rd(d, popt_ltheta[0], popt_ltheta[1])) / (errori_periodi))**2
print("FIT l E theta PARAMETRI\n")
print("l ottimale : {}+-{}\n".format(popt_ltheta[0], np.diagonal(pcov_ltheta)[0]))
print("theta ottimale : {}+-{}\n".format(popt_ltheta[1], np.diagonal(pcov_ltheta)[1]))
print("Chi2 singolo: {}\n".format(chi2))
print("Chi2 complessivo: {}\n".format(np.sum(chi2)))

# Fit eliminando il secondo punto
medie_periodi_new = np.delete(medie_periodi, 1)
errori_periodi_new = np.delete(errori_periodi, 1)
d_new = np.delete(d, 1)
[popt, pcov] = scipy.optimize.curve_fit(modello, d_new, medie_periodi_new, [1], errori_periodi_new)
chi2 = ((medie_periodi_new-modello(d_new, popt[0])) / (errori_periodi_new))**2
print("Chi2 senza secondo punto: ", chi2, "\n")
print(popt, "\n")
print("Chi2 senza secondo punto complessivo: ",np.sum(chi2), "\n\n")

# Confronto y con derivata
derivata = derivata_modello(d, 1.05)
print("Derivata:\n{}\n<<\n{}".format(derivata*errore_d, errori_periodi))
print("Derivata punto per punto:",derivata, "\n\n")

# Ridefinizione errori ultimi 2 errori e nuovo fit
errori_periodi[3] = np.sqrt(errori_periodi[3]**2 + (derivata[3]*errore_d[3])**2)
errori_periodi[4] = np.sqrt(errori_periodi[4]**2 + (derivata[4]*errore_d[4])**2)
errori_periodi[1] = np.sqrt(errori_periodi[1]**2 + (derivata[1]*errore_d[1])**2)
[popt_lnew, pcov_lnew] = scipy.optimize.curve_fit(modello, d, medie_periodi, [1], errori_periodi)
chi2 = ((medie_periodi-modello(d, popt_lnew[0])) / (errori_periodi))**2
pvalue = 1 - scipy.stats.chi2.cdf(np.sum(chi2), 4)
print("FIT CON NUOVI ERRORI\n")
print("Chi2 singolo: {}\n".format(chi2))
print("Chi2 complessivo: {}\n".format(np.sum(chi2)))
print("Pvalue: {}\n\n".format(pvalue))


# Parte grafici
# Inizializzazione array
x_t = np.linspace(50, 5, 2000) / 100
y_t = modello(x_t, 1.05)
y_t_opt = modello(x_t, popt_l[0])

# Grafico con modello semplice
plt.figure("T semplici")
plt.errorbar(d, medie_periodi, errori_periodi, errore_d, fmt = 'o', label = "T misurati")
plt.plot(x_t, y_t, label = "Modello")
plt.xlabel("[m]")
plt.ylabel("[s]")
plt.title("Periodi modello semplice")
plt.legend()
plt.show()

# Grafico con modello +- sigma
plt.figure("T +- sigma")
plt.errorbar(d, medie_periodi, errori_periodi, errore_d, fmt = 'o', label = "T misurati")
plt.plot(x_t, y_t, label = "Modello")
plt.plot(x_t, modello(x_t, 1.052), label = "Modello +sigma")
plt.plot(x_t, modello(x_t, 1.048), label = "Modello -sigma")
plt.xlabel("[m]")
plt.ylabel("[s]")
plt.title("Periodi modello +- sigma")
plt.legend()
plt.show()

# Grafico con parametri ottimali
plt.figure("T opt")
plt.errorbar(d, medie_periodi, errori_periodi, errore_d, fmt = 'o', label = "T misurati")
plt.plot(x_t, modello(x_t, popt_l[0]), label = "Modello ottimale")
plt.xlabel("[m]")
plt.ylabel("[s]")
plt.title("Periodi con parametri ottimali")
plt.legend()
plt.show()

# Residui con parametri ottimali
plt.figure("Residui T opt")
plt.errorbar(d, medie_periodi-modello(d, popt_l[0]), errori_periodi, errore_d, fmt = 'o', label = "T misurati")
plt.plot(x_t, x_t*0, label = "Modello ottimale")
plt.xlabel("[m]")
plt.ylabel("[s]")
plt.title("Residui con parametri ottimali")
plt.grid()
plt.legend()
plt.show()