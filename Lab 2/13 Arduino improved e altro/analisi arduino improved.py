import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import uncertainties as unc
from uncertainties import unumpy

def funzione (t, A, tau, w, phi, Vbias) :
    return A * np.exp(-t/tau) * np.cos(w*t+phi) + Vbias

def derivata (t, A, tau, w, phi, Vbias) :
    return -A * np.exp(-t/tau) * (np.cos(w*t+phi)/tau + w*np.sin(w*t+phi))

def chi_quadro (y, dy, t, popt) :
    return np.sum(((y-funzione(t, *popt)) / dy)**2)
    
    
C = unumpy.uarray(0.1e-6, 0.01e-6)
tempo, dt, Vc, dVc = np.genfromtxt("alluminio pieno.txt", unpack = True, skip_header=1, skip_footer = 14)
tempo = tempo*10**-6
dt = dt*10**-6
dVc_eff = dVc
p0 = (300, 0.02, 2000, np.pi/2, 489)

for i in range(5) :
    popt, pcov = scipy.optimize.curve_fit(funzione, tempo, Vc, p0, dVc_eff, absolute_sigma = True)
    chi_2 = chi_quadro(Vc, dVc, tempo, popt)
    dVc_eff = np.sqrt(dVc**2 + (dt*derivata(tempo, *popt))**2)

a = ["A", "tau", "w", "phi", "Vbias"]
errore = np.sqrt(np.diag(pcov))
print("chi2 rid =", chi_2/237)
for i in range (len(a)) :
    print(a[i] + "\t" + str(popt[i]) + " +- " + str(errore[i]))
    
tau = unumpy.uarray(popt[1], errore[1])
w = unumpy.uarray(popt[2], errore[2])
w0_quad = w**2 + 1/tau**2    
L = 1/(C*w0_quad)
print("T =", 2*np.pi/w)
print("L =", L)
print("r =", 2*L/tau)
print("Qef =", w*tau/2)


t = np.linspace(0, 0.016, 4000)
plt.figure()
plt.subplot(211)
plt.title("Ferro laminato")
plt.ylabel("ddp [digit]")
plt.xlabel("t [ms]")
plt.errorbar(tempo, Vc, dVc, dt, fmt = '.', label = "Data")
plt.plot(t, funzione(t,*popt), label = "Fit")
plt.legend()

plt.subplot(212)
plt.title("Residui normalizzati")
plt.xlabel("t [ms]")
plt.errorbar(tempo, (Vc-funzione(tempo, *popt))/dVc, fmt=".")
plt.plot(t, t*0)
plt.show()