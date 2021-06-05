import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.odr import odrpack


# Funzione di fit
def linear(b, x) :
    return b[0]*x + b[1]

def retta(x, a, b) :
    return a*x + b

# Lettura dati
datix = np.array([1, 2, 3, 4, 5])
datiy = np.array([2, 4.2, 6.1, 7.8, 9.95])
sigmay = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
sigmax = np.array([0.1, 0.1, 0.1, 0.1, 0.1])*0.5

# Fit odr
ddof = 2
model = odrpack.Model(linear)
data = odrpack.RealData(datix, datiy, sx=sigmax, sy=sigmay)
odr = odrpack.ODR(data, model, beta0=(1., 1.))
out = odr.run()
popt_linear, pcov_linear = out.beta, out.cov_beta
chi2_linear = out.sum_square

print("Parametri ottimali: {}\n".format(popt_linear))
print("Errori parametri: {}\n".format(np.sqrt(np.diagonal(pcov_linear))))
print("Chi2: {}, aspettato {}\n".format(chi2_linear, len(datix)-ddof))

# Parte grafici
x_plot = np.linspace(0, 10, 1000)
plt.figure()
plt.title("Grafico lineare")
plt.errorbar(datix, datiy, sigmay, sigmax, fmt = 'o')
plt.plot(x_plot, retta(x_plot, *popt_linear))
plt.grid()
plt.ylabel('Tempo [s]')
plt.xlabel('Distanza [cm]')
# plt.savefig("Fit con odr.png", bbox_inches='tight')
plt.show()