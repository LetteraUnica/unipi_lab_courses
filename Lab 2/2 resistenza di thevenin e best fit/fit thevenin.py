import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# Metodo con fit
def funzione(x, v0, Rg) :
    return v0/(x+Rg)
    
def derivata(x, V0, Rg) :
    return -V0/(Rg+x)**2

def chi2(x, popt, y, dy) :
    return np.sum(((y-funzione(x, *popt))/dy)**2)


# Dati
Rj, dRj, Ra, dRa = np.genfromtxt("resistenze.txt", skip_header=1, unpack=True)
Ij, dIj = np.genfromtxt("correnti.txt", skip_header=1, unpack=True)
Ij /= 1000
dIj /= 1000
Rx = Rj+Ra
dRx = (dRj**2+dRa**2)**0.5


# Fit
ndof = 10
dI_eff = dIj
popt, pcov = scipy.optimize.curve_fit(funzione, Rx, Ij, [4, 18], dI_eff, absolute_sigma=False)
chi2_new = chi2(Rx, popt, Ij, dI_eff)
chi2_old = -1
print(chi2_new)
for i in range (10) :
    dI_eff = ((derivata(Rx, *popt)*dRx)**2 + dIj**2)**0.5
    popt, pcov = scipy.optimize.curve_fit(funzione, Rx, Ij, [4, 18], dI_eff, absolute_sigma=False)
    chi2_old = chi2_new
    chi2_new = chi2(Rx, popt, Ij, dI_eff)
    print(chi2_new)

print("\nV0 ", popt[0], pcov[0][0]**0.5)
print("Rg ", popt[1], pcov[1][1]**0.5)
print("chi2 ", chi2_new)
print("chi2 ridotto {}/{}={}".format(chi2_new, ndof, chi2_new/ndof))
print("cov normalizzata", pcov[1][0]/(pcov[0][0]*pcov[1][1])**0.5, "\n")


# Grafici
# Plot funzione
x = np.linspace(1, 6, 4000)
x = 10**x
plt.figure()
plt.subplot(211)
plt.xscale('log')
plt.yscale('log')
plt.errorbar(Rx, Ij, yerr=dIj, xerr=dRx, fmt='.', label="Misure")
plt.plot(x, funzione(x, *popt), label="Fit")
plt.xlabel("R_eq [ohm]")
plt.ylabel("I [A]")
plt.legend()

# Plot residui normalizzati
plt.subplot(212)
plt.xscale('log')
plt.errorbar(Rx, (Ij-funzione(Rx, *popt))/dI_eff, fmt='o', label="Residui normalizzati")
plt.plot(x, x*0)
plt.xlabel("R_eq [ohm]")
plt.legend()
plt.show()