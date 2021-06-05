import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.odr import odrpack

def modello_fit(x, a, b):
    return a*x + b
    
def modello(B, x):
    return B[0]*x + B[1]

# Lettura dati
p_convergente, q_convergente = np.genfromtxt("C:/Users/Lorenzo/Desktop/Relazioni fisica/11) Focali/convergente.txt", skip_header = 2, unpack = True)
p_divergente, q_divergente = np.genfromtxt("C:/Users/Lorenzo/Desktop/Relazioni fisica/11) Focali/divergente.txt", skip_header = 2, unpack = True)

p_convergente = np.delete(p_convergente, 5)*0.01    # Elimino il punto che esce
q_convergente =  np.delete(q_convergente, 5)*0.01   # Elimino il punto che esce
errore_convergente_p = np.ones(len(p_convergente))*0.2*0.01
errore_convergente_q = np.ones(len(p_convergente))*0.5*0.01

p_divergente = p_divergente*0.01
q_divergente = q_divergente*0.01
errore_divergente_p = np.ones(len(p_divergente))*0.2*0.01
errore_divergente_q = np.ones(len(p_divergente))*0.5*0.01


# Fit convergente
model = odrpack.Model(modello)
data = odrpack.RealData(1/p_convergente, 1/q_convergente, sx = errore_convergente_p/(p_convergente**2), sy = errore_convergente_q/(q_convergente**2))
odr = odrpack.ODR(data, model, beta0=(1, 1))
out = odr.run()
popt_convergente, pcov_convergente = out.beta, out.cov_beta
chi2_convergente = out.sum_square
print("Parametri ottimali: {}".format(popt_convergente))
print("Errori parametri: {}".format(np.sqrt(np.diagonal(pcov_convergente))))
print("Chi2: {}, aspettato {}\n\n".format(chi2_convergente, len(p_convergente)-2))

# Fit divergente
model = odrpack.Model(modello)
data = odrpack.RealData(1/p_divergente, 1/q_divergente, sx = errore_divergente_p/(p_divergente**2), sy = errore_divergente_q/(q_divergente**2))
odr = odrpack.ODR(data, model, beta0=(1, 1))
out = odr.run()
popt_divergente, pcov_divergente = out.beta, out.cov_beta
chi2_divergente = out.sum_square
print("Parametri ottimali: {}" .format(popt_divergente))
print("Errori parametri: {}" .format(np.sqrt(np.diagonal(pcov_divergente))))
print("Chi2: {}, aspettato {}\n\n" .format(chi2_divergente, len(p_divergente)-2))


# Plot convergente
x = np.linspace(1, 4, 4000)
plt.figure("Convergente")
plt.subplot(211)
plt.errorbar(1/p_convergente, 1/q_convergente, xerr = errore_convergente_p/(p_convergente**2), yerr = errore_convergente_q/(q_convergente**2), fmt = 'o')
plt.plot(x, modello_fit(x, *popt_convergente))

plt.subplot(212)
plt.errorbar(1/p_convergente, (1/q_convergente)-modello_fit(1/p_convergente, *popt_convergente), xerr = errore_convergente_p/(p_convergente**2), yerr = errore_convergente_q/(q_convergente**2), fmt = 'o')
plt.plot(x, x*0)
plt.show()

# Plot divergente
x = np.linspace(5, 30, 4000)
plt.figure("Divergente")
plt.subplot(211)
plt.errorbar(1/p_divergente, 1/q_divergente, xerr = errore_divergente_p/(p_divergente**2), yerr = errore_divergente_q/(q_divergente**2), fmt = 'o')
plt.plot(x, modello_fit(x, *popt_divergente))

plt.subplot(212)
plt.errorbar(1/p_divergente, (1/q_divergente)-modello_fit(1/p_divergente, *popt_divergente), xerr = errore_divergente_p/(p_divergente**2), yerr = errore_divergente_q/(q_divergente**2), fmt = 'o')
plt.plot(x, x*0)
plt.show()