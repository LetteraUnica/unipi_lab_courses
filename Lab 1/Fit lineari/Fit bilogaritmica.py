import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

# Funzione di fit
def linear(x, a, b) :
    return a*x + b


# Lettura dati
datix = np.array([4.755, 6.245, 7.135, 7.93, 9.125])*0.5
datiy = np.array([3.523, 8.357, 11.892, 16.321, 24.84])
sigmay = np.array([0.1, 0.1, 0.1, 0.1, 0.1])*0.3
sigmax = np.array([0.1, 0.1, 0.1, 0.1, 0.1])*0.1*0.5

sigmay = abs(np.log(datiy-sigmay)-np.log(datiy))
sigmax = abs(np.log(datix-sigmax)-np.log(datix))
datiy = np.log(datiy)
datix = np.log(datix)

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
plt.errorbar(datix, datiy, sigmay, sigmax, fmt='o')
plt.plot(x_fit, linear(x_fit, *popt_lineare))
plt.subplot(212)
plt.errorbar(datix, datiy-linear(datix, *popt_lineare), sigmay, fmt='o')
plt.plot(x_fit, x_fit*0)
plt.savefig("Cilindro non isolato.png", bbox_inches='tight')
plt.show()