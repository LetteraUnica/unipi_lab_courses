import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# Metodo con fit
def funzione(x, a, b) :
    return a + b*x
    
def derivata(x, a, b) :
    return b

def chi2(x, popt, y, dy) :
    return np.sum(((y-funzione(x, *popt))/dy)**2)


# Dati
Rj, dRj = np.genfromtxt("digitalizzate.txt", skip_header=0, unpack=True)
Ij, dIj = np.genfromtxt("ddp.txt", skip_header=0, unpack=True)

# Fit
ndof = 10
dI_eff = dIj
popt, pcov = scipy.optimize.curve_fit(funzione, Rj, Ij, [4, 18], dI_eff, absolute_sigma=False)
chi2_new = chi2(Rj, popt, Ij, dI_eff)
chi2_old = -1
print(chi2_new)
for i in range (10) :
    dI_eff = ((derivata(Rj, *popt)*dRj)**2 + dIj**2)**0.5
    popt, pcov = scipy.optimize.curve_fit(funzione, Rj, Ij, [4, 18], dI_eff, absolute_sigma=False)
    chi2_old = chi2_new
    chi2_new = chi2(Rj, popt, Ij, dI_eff)
    print(chi2_new)

print("\nintercetta", popt[0], pcov[0][0]**0.5)
print("coeff angolare", popt[1], pcov[1][1]**0.5)
print("chi2 ", chi2_new)
print("chi2/ndof {}/{}={}".format(chi2_new, ndof, chi2_new/ndof))
print("cov normalizzata", pcov[1][0]/(pcov[0][0]*pcov[1][1])**0.5, "\n")


# Grafici
# Plot funzione
x = np.linspace(0, 1000, 4000)
plt.figure()
plt.subplot(211)
plt.errorbar(Rj, Ij, yerr=dIj, xerr=dRj, fmt='.', label="Misure")
plt.plot(x, funzione(x, *popt), label="Fit")
plt.xlabel('Valori digitalizzati X [digit]')
plt.ylabel('ddp [V]')
plt.legend()

# Plot residui normalizzati
plt.subplot(212)
plt.errorbar(Rj, (Ij-funzione(Rj, *popt))/dI_eff, fmt='.', label="Residui normalizzati")
plt.plot(x, x*0)
plt.xlabel('Valori digitalizzati X [digit]')
plt.legend()
plt.show()