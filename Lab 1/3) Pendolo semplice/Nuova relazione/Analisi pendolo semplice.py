import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

# Funzione di fit
def linear(x, a, b) :
    return a*x + b


# Lettura dati
datix = np.array([1, 2, 3, 4, 5])
datiy = np.array([2, 4.2, 6.1, 7.8, 9.95])
sigmay = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
sigmax = np.array([0.1, 0.1, 0.1, 0.1, 0.1])*0.1

# Fit lineare
ddof = 2
popt_lineare, pcov_lineare = scipy.optimize.curve_fit(linear, datix, datiy, [1, 1], sigmay, absolute_sigma = False)
chi2_lineare = ((datiy - linear(datix, *popt_lineare)) / (sigmay))**2

print("Parametri ottimali: {}\n".format(popt_lineare))
print("Errori parametri: {}\n".format(np.sqrt(np.diagonal(pcov_lineare))))
print("Chi2: {}, aspettato {}\n".format(chi2_lineare.sum(), len(chi2_lineare)-ddof))

# Grafici
# Inserire legenda
x_fit = np.linspace(0, 6, 1000)
plt.figure()
plt.subplot(211)
plt.xlabel("Grandezza1 [cm]")
plt.ylabel("Grandezza2 [cm]")
plt.errorbar(datix, datiy, sigmay, sigmax, fmt='o')
plt.plot(x_fit, linear(x_fit, *popt_lineare))
plt.subplot(212)
plt.ylabel("Grandezza2 [cm]")
plt.errorbar(datix, datiy-linear(datix, *popt_lineare), sigmay, fmt='o')
plt.plot(x_fit, x_fit*0)
plt.savefig("Pendolo semplice.png", bbox_inches='tight')
plt.show()