import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

# Funzione di fit
def funzione(x, a, b) :
    return a*x + b

# Derivata
def derivata(x, popt) :
    h = 10**-4
    return (funzione(x+h, *popt) - funzione(x-h, *popt)) / (2*h)

# Lettura dati
np.random.seed(4)
datix = np.linspace(0, 10, 100) + np.random.normal(0.1, 0.2, 100)
datiy = np.linspace(0, 40, 100) + np.random.normal(0.1, 0.15, 100)
sigmay = np.full(100, 0.1)
sigmax = np.full(100, 0.2)

# Fit funzione, mq modificati
ddof = 2
chi2_new, i = -1, 0
chi2_old = 0
n_dati = len(datix)
dxy = sigmay

while (abs(chi2_new - chi2_old) > 10**(-3)) :
    chi2_old = chi2_new
    popt_funzione, pcov_funzione = scipy.optimize.curve_fit(funzione, datix, datiy, [1, 1], dxy, absolute_sigma = False)
    chi2_new = np.sum(((datiy - funzione(datix, *popt_funzione)) / (dxy))**2)
    
    dxy = np.sqrt(sigmay**2 + (derivata(datix, popt_funzione) * sigmax)**2)
    
    print("Passo ", i)
    print("Parametri ottimali: {}".format(popt_funzione))
    print("Errori parametri: {}".format(np.sqrt(np.diagonal(pcov_funzione))))
    print("Chi2: {}, aspettato {}".format(chi2_new, n_dati-ddof))
    print("\n")
    i += 1

# Grafici
# Inserire legenda
x_fit = np.linspace(-1, 15, 1000)
plt.figure()
plt.subplot(211)
plt.xlabel("Grandezza1 [cm]")
plt.ylabel("Grandezza2 [cm]")
plt.errorbar(datix, datiy, sigmay, sigmax, fmt='o')
plt.plot(x_fit, funzione(x_fit, *popt_funzione))
plt.subplot(212)
plt.ylabel("Grandezza2 [cm]")
plt.errorbar(datix, datiy-funzione(datix, *popt_funzione), dxy, fmt='o')
plt.plot(x_fit, x_fit*0)
plt.savefig("Cilindro non isolato.png", bbox_inches='tight')
plt.show()