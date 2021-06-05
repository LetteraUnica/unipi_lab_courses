import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

# carica_t di fit
def carica_t(t, a, tau) :
    return a*(1-np.exp(-t/tau))

# scarica_t di fit
def scarica_t(t, a, b, tau) :
    return a*np.exp(-t/tau)+b

# Derivata
def derivata_c(x, popt) :
    h = 10**-4
    return (carica_t(x+h, *popt) - carica_t(x-h, *popt)) / (2*h)

def derivata_s(x, popt) :
    h = 10**-4
    return (scarica_t(x+h, *popt) - scarica_t(x-h, *popt)) / (2*h)

# Lettura dati, Da cambiare
tempo_carica, ddp_carica = np.genfromtxt("condensatore2tau_C.txt", unpack=True)
tempo_scarica, ddp_scarica = np.genfromtxt("condensatore2tau_S.txt", unpack=True)


# Fit carica_t, minimi quadrati modificati
ddof = 2
chi2_new, chi2_old = -1, 0
n_dati = len(tempo_carica) # Da cambiare
datix, datiy = tempo_carica, ddp_carica
sigmax = np.ones(n_dati)*4
sigmay = np.ones(n_dati)

dxy = sigmay

i = 0
while (abs(chi2_new - chi2_old) > 10**(-3)) :
    chi2_old = chi2_new
    popt_carica_t, pcov_carica_t = scipy.optimize.curve_fit(carica_t, datix, datiy, [1, 1], dxy, absolute_sigma = False)
    chi2_new = np.sum(((datiy - carica_t(datix, *popt_carica_t)) / (dxy))**2)

    dxy = np.sqrt(sigmay**2 + (derivata_c(datix, popt_carica_t) * sigmax)**2)
    print("Chi2: {}, aspettato {}".format(chi2_new, n_dati-ddof))
    i += 1

sigma_carica = dxy
print("\n")
print("Parametri ottimali: {}".format(popt_carica_t))
print("Errori parametri: {}".format(np.sqrt(np.diagonal(pcov_carica_t))))
print("Chi2: {}, aspettato {}".format(chi2_new, n_dati-ddof))
print("Cov normalizzata", pcov_carica_t[1][0]/(pcov_carica_t[0][0]*pcov_carica_t[1][1])**0.5, "\n")

# Fit scarica_t, minimi quadrati modificati
ddof = 3
chi2_new, chi2_old = -1, 0
sigmax = tempo_carica*0
sigmay = np.ones(n_dati)
datix, datiy = tempo_scarica, ddp_scarica
n_dati = len(datix) # Da cambiare
dxy = sigmay

while (abs(chi2_new - chi2_old) > 10**(-3)) :
    chi2_old = chi2_new
    popt_scarica_t, pcov_scarica_t = scipy.optimize.curve_fit(scarica_t, datix, datiy, [1000, 1, 1], dxy, absolute_sigma = False)
    chi2_new = np.sum(((datiy - scarica_t(datix, *popt_scarica_t)) / (dxy))**2)

    dxy = np.sqrt(sigmay**2 + (derivata_s(datix, popt_scarica_t) * sigmax)**2)
    print("Chi2: {}, aspettato {}".format(chi2_new, n_dati-ddof))
    i += 1

sigma_scarica = dxy
print("\n")

print("Parametri ottimali: {}".format(popt_scarica_t))
print("Errori parametri: {}".format(np.sqrt(np.diagonal(pcov_scarica_t))))
print("Chi2: {}, aspettato {}".format(chi2_new, n_dati-ddof))


# Grafici
x_fit = np.linspace(0, 150000, 4000)
# Carica
datix, datiy = tempo_carica, ddp_carica
plt.figure("Carica")
plt.subplot(211)
plt.grid()
plt.ylabel("ddp [digit]")
plt.xlabel("Tempo [us]")
plt.errorbar(datix, datiy, sigmay, sigmax, fmt='.', label = "Dati")
plt.plot(x_fit, carica_t(x_fit, *popt_carica_t), label = "Fit")
plt.legend()

plt.subplot(212)
plt.grid()
plt.xlabel("tempo [us]")
plt.errorbar(datix, (datiy-carica_t(datix, *popt_carica_t))/sigma_carica, fmt='.')
plt.plot(x_fit, x_fit*0)
plt.savefig("Cilindro non isolato.png", bbox_inches='tight')
plt.show()

# Scarica
datix, datiy = tempo_scarica, ddp_scarica
plt.figure("Scarica")
plt.subplot(211)
plt.ylabel("ddp [digit]")
plt.xlabel("Tempo [us]")
plt.errorbar(datix, datiy, sigmay, sigmax, fmt='.', label = "Dati")
plt.plot(x_fit, scarica_t(x_fit, *popt_scarica_t), label = "Fit")
plt.legend()

plt.subplot(212)
plt.xlabel("Tempo [us]")
plt.errorbar(datix, (datiy-scarica_t(datix, *popt_scarica_t))/sigma_scarica, fmt='.')
plt.plot(x_fit, x_fit*0)
plt.show()