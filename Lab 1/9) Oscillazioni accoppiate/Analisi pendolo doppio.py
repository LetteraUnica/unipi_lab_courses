import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def taglia_dati(t, x, min, max) :
    mask = (t>min) * (t<max)
    return t[mask], x[mask]

def fit_oscillazioni_singole(t, A, w, phi, damp, trasl) :
    return np.e**(-damp*t) * A * np.sin(w*t + phi) + trasl

def fit_battimenti(t, A, wf, wc, phi1, phi2, damp, trasl) :
    return A*np.e**(-damp*t) * (np.cos(wf*t+phi1) + np.cos(wc*t+phi2)) + trasl

def fit_prostaferesi(t, A, wp, wb, phi_sum, phi_min, damp, trasl) :
    return 2*A*np.e**(-damp*t) * (np.cos(wp*t + 0.5*phi_sum) * np.cos(wb*t + 0.5*phi_min)) + trasl


# Lettura dati
oscillazione_singola = np.genfromtxt("C:/Users/Lorenzo/Desktop/Lab fisica/9) Oscillazioni Accoppiate/Dati non modificati/Oscillazione1.txt", skip_header = 4, unpack = True)
oscillazione_dampata = np.genfromtxt("C:/Users/Lorenzo/Desktop/Lab fisica/9) Oscillazioni Accoppiate/Dati/OscillazioneDampata1.txt", skip_header = 4, unpack = True)
oscillazione_infase = np.genfromtxt("C:/Users/Lorenzo/Desktop/Lab fisica/9) Oscillazioni Accoppiate/Dati/PendoliInFase1.txt", skip_header = 4, unpack = True)
oscillazione_controfase = np.genfromtxt("C:/Users/Lorenzo/Desktop/Lab fisica/9) Oscillazioni Accoppiate/Dati/OscillazioneControfase1.txt", skip_header = 4, unpack = True)
battimenti = np.genfromtxt("C:/Users/Lorenzo/Desktop/Lab fisica/9) Oscillazioni Accoppiate/Dati/Battimenti3.txt", skip_header = 4, unpack = True)


# Analisi dati
# Traslazione punti
oscillazione_singola[1] += -np.average(oscillazione_singola[1])
oscillazione_dampata[1] += -np.average(oscillazione_dampata[1])
oscillazione_infase[1] += -np.average(oscillazione_infase[1])
oscillazione_infase[3] += -np.average(oscillazione_infase[3])
oscillazione_controfase[1] += -np.average(oscillazione_controfase[1])
oscillazione_controfase[3] += -np.average(oscillazione_controfase[3])
battimenti[1] += -np.average(battimenti[1])
battimenti[3] += -np.average(battimenti[3])

# Fit oscillazioni singole
[popt_osc_singola, pcov_osc_singola] = scipy.optimize.curve_fit(fit_oscillazioni_singole, oscillazione_singola[0], oscillazione_singola[1], [110, 4.2, -np.pi, 0.02, 1])
print("OSCILLAZIONE SEMPLICE\n")
print("Parametri: {}\n".format(popt_osc_singola))
print("Errori: {}\n\n".format(np.diagonal(pcov_osc_singola)**0.5))

# Fit oscillazioni dampate
[popt_osc_dampata, pcov_osc_dampata] = scipy.optimize.curve_fit(fit_oscillazioni_singole, oscillazione_dampata[0], oscillazione_dampata[1], [110, 4.2, -np.pi, 0.02, 1])
print("OSCILLAZIONE DAMPATA\n")
print("Parametri: {}\n".format(popt_osc_dampata))
print("Errori: {}\n\n".format(np.diagonal(pcov_osc_dampata)**0.5))

# Fit oscillazioni accoppiate in fase
ta_infase, xa_infase, tb_infase, xb_infase = oscillazione_infase
ta_infase, xa_infase = taglia_dati(ta_infase, xa_infase, 8, 15)
tb_infase, xb_infase = taglia_dati(tb_infase, xb_infase, 8, 15)
[popt_infase1, pcov_infase1] = scipy.optimize.curve_fit(fit_oscillazioni_singole, ta_infase, xa_infase, [110, 4.2, -np.pi, 0.02, 1])
[popt_infase2, pcov_infase2] = scipy.optimize.curve_fit(fit_oscillazioni_singole, tb_infase, xb_infase, [110, 4.2, -np.pi, 0.02, 1])
print("OSCILLAZIONE IN FASE\n")
print("Parametri: {}\n".format(popt_infase1))
print("Errori: {}\n".format(np.diagonal(pcov_infase1)**0.5))
print("Parametri: {}\n".format(popt_infase2))
print("Errori: {}\n\n".format(np.diagonal(pcov_infase2)**0.5))

# Fit oscillazioni accoppiate controfase
# ta_controfase, xa_controfase, tb_controfase, xb_controfase = oscillazione_controfasefase
# ta_controfase, xa_controfase = taglia_dati(ta_controfase, xa_controfase, 8, 15)
# tb_controfase, xb_controfase = taglia_dati(tb_controfase, xb_controfase, 8, 15)
[popt_controfase1, pcov_controfase1] = scipy.optimize.curve_fit(fit_oscillazioni_singole, oscillazione_controfase[0], oscillazione_controfase[1], [110, 4.2, -np.pi, 0.02, 1])
[popt_controfase2, pcov_controfase2] = scipy.optimize.curve_fit(fit_oscillazioni_singole, oscillazione_controfase[2], oscillazione_controfase[3], [110, 4.2, -np.pi, 0.02, 1])
print("OSCILLAZIONE CONTROFASE\n")
print("Parametri: {}\n".format(popt_controfase1))
print("Errori: {}\n".format(np.diagonal(pcov_controfase1)**0.5))
print("Parametri: {}\n".format(popt_controfase2))
print("Errori: {}\n\n".format(np.diagonal(pcov_controfase2)**0.5))

# Fit battimenti
[popt_battimenti1, pcov_battimenti1] = scipy.optimize.curve_fit(fit_battimenti, battimenti[0], battimenti[1], [110, 4.6, 4.4, 1, 1, 0.01, 1])
[popt_battimenti2, pcov_battimenti2] = scipy.optimize.curve_fit(fit_battimenti, battimenti[2], battimenti[3], [150, 4.4, 4.6, 0, 0, 0.01, 1])
print("BATTIMENTI\n")
print("Parametri: {}\n".format(popt_battimenti1))
print("Errori: {}\n".format(np.diagonal(pcov_battimenti1)**0.5))
print("Parametri: {}\n".format(popt_battimenti2))
print("Errori: {}\n\n".format(np.diagonal(pcov_battimenti2)**0.5))

# Fit battimenti con prostaferesi
[popt_prost1, pcov_prost1] = scipy.optimize.curve_fit(fit_prostaferesi, battimenti[0], battimenti[1], [80, 0.08, 4.6, 1, 1, 0.01, 1])
[popt_prost2, pcov_prost2] = scipy.optimize.curve_fit(fit_prostaferesi, battimenti[2], battimenti[3], [60, 0.06, 4.5, 1, 1, 0.01, 1])
print("BATTIMENTI PROSTAFERESI\n")
print("Parametri: {}\n".format(popt_prost1))
print("Errori: {}\n".format(np.diagonal(pcov_prost1)**0.5))
print("Parametri: {}\n".format(popt_prost2))
print("Errori: {}\n\n".format(np.diagonal(pcov_prost2)**0.5))


# Parte grafici
# Oscillazione semplice
x = np.linspace(oscillazione_singola[0,0]-5, oscillazione_singola[0,-1]+5, 4000)
plt.figure("Osc. semplice")
plt.subplot(211)
plt.errorbar(oscillazione_singola[0], oscillazione_singola[1], yerr = 1, label = "Punti", fmt = 'o')
plt.plot(x, fit_oscillazioni_singole(x, *popt_osc_singola), label = "Modello")
plt.xlabel("[s]")
plt.ylabel("[ua]")
plt.title("Grafico oscillazione semplice")
plt.legend()

# Residui
plt.subplot(212)
plt.errorbar(oscillazione_singola[0], oscillazione_singola[1]-fit_oscillazioni_singole(oscillazione_singola[0], *popt_osc_singola), yerr = 1, label = "Punti", fmt = 'o')
plt.plot(x, x*0, label = "Modello")
plt.title("Residui")
plt.show()


# Oscillazione dampata
x = np.linspace(oscillazione_dampata[0,0]-5, oscillazione_dampata[0,-1]+5, 4000)
plt.figure("Osc. dampata")
plt.subplot(211)
plt.errorbar(oscillazione_dampata[0], oscillazione_dampata[1], yerr = 1, label = "Punti", fmt = 'o')
plt.plot(x, fit_oscillazioni_singole(x, *popt_osc_dampata), label = "Modello")
plt.xlabel("[s]")
plt.ylabel("[ua]")
plt.title("Grafico oscillazione dampata")
plt.legend()

# Residui
plt.subplot(212)
plt.errorbar(oscillazione_dampata[0], oscillazione_dampata[1]-fit_oscillazioni_singole(oscillazione_dampata[0], *popt_osc_dampata), yerr = 1, label = "Punti", fmt = 'o')
plt.plot(x, x*0, label = "Modello")
plt.title("Residui")
plt.show()


# Oscillazione in fase
x = np.linspace(ta_infase[0]-5, ta_infase[-1]+5, 4000)
plt.figure("Osc. in fase")
plt.errorbar(ta_infase, xa_infase, yerr = 1, label = "Punti p1", fmt = 'o', color = "blue", markersize=5)
plt.plot(x, fit_oscillazioni_singole(x, *popt_infase1), label = "Modello p1", color = "cyan")
plt.errorbar(tb_infase, xb_infase, yerr = 1, label = "Punti p2", fmt = 'o', color = "red", markersize=5)
plt.plot(x, fit_oscillazioni_singole(x, *popt_infase2), label = "Modello p2", color = "orange")
plt.xlabel("[s]")
plt.ylabel("[ua]")
plt.title("Grafico oscillazione in fase")
plt.legend()
plt.show()

# Residui
plt.figure("R. osc. in fase")
plt.subplot(211)
plt.errorbar(ta_infase, xa_infase-fit_oscillazioni_singole(ta_infase, *popt_infase1), yerr = 1, fmt = 'o')
plt.plot(x, x*0, label = "Modello")
plt.title("Residui in fase p1")
plt.subplot(212)
plt.errorbar(tb_infase, xb_infase-fit_oscillazioni_singole(tb_infase, *popt_infase2), yerr = 1,  fmt = 'o')
plt.plot(x, x*0, label = "Modello")
plt.title("Residui in fase p2")
plt.show()


# Oscillazione controfase
x = np.linspace(oscillazione_controfase[0,0]-5, oscillazione_controfase[0,-1]+5, 4000)
plt.figure("Osc. controfase")
plt.errorbar(oscillazione_controfase[0], oscillazione_controfase[1], yerr = 1, label = "Punti p1", fmt = 'o', color = "blue", markersize=5)
plt.plot(x, fit_oscillazioni_singole(x, *popt_controfase1), label = "Modello p1", color = "cyan")
plt.errorbar(oscillazione_controfase[2], oscillazione_controfase[3], yerr = 1, label = "Punti p2", fmt = 'o', color = "red", markersize=5)
plt.plot(x, fit_oscillazioni_singole(x, *popt_controfase2), label = "Modello p2", color = "orange")
plt.xlabel("[s]")
plt.ylabel("[ua]")
plt.title("Grafico oscillazione controfase")
plt.legend()
plt.show()

# Residui
plt.figure("R. osc. controfase")
plt.subplot(211)
plt.errorbar(oscillazione_controfase[0], oscillazione_controfase[1]-fit_oscillazioni_singole(oscillazione_controfase[0], *popt_controfase1), yerr = 1, fmt = 'o')
plt.plot(x, x*0, label = "Modello")
plt.title("Residui controfase p1")
plt.subplot(212)
plt.errorbar(oscillazione_controfase[2], oscillazione_controfase[3]-fit_oscillazioni_singole(oscillazione_controfase[2], *popt_controfase2) ,yerr = 1, fmt = 'o')
plt.plot(x, x*0, label = "Modello")
plt.title("Residui controfase p2")
plt.show()


# Battimenti
x = np.linspace(battimenti[0,0]-5, battimenti[0,-1]+5, 4000)
plt.figure("Battimenti")
plt.errorbar(battimenti[0], battimenti[1], yerr = 1, label = "Punti p1", fmt = 'o', color = "blue", markersize=5)
plt.plot(x, fit_battimenti(x, *popt_battimenti1), label = "Modello p1", color = "cyan")
plt.errorbar(battimenti[2], battimenti[3], yerr = 1, label = "Punti p2", fmt = 'o', color = "red", markersize=5)
plt.plot(x, fit_battimenti(x, *popt_battimenti2), label = "Modello p2", color = "orange")
plt.xlabel("[s]")
plt.ylabel("[ua]")
plt.title("Grafico battimenti")
plt.legend()
plt.show()

# Residui
plt.figure("R. battimenti")
plt.subplot(211)
plt.errorbar(battimenti[0], battimenti[1]-fit_battimenti(battimenti[0], *popt_battimenti1), yerr = 1, fmt = 'o')
plt.plot(x, x*0, label = "Modello")
plt.title("Residui battimenti p1")
plt.subplot(212)
plt.errorbar(battimenti[2], battimenti[3]-fit_battimenti(battimenti[2], *popt_battimenti2), yerr = 1, fmt = 'o')
plt.plot(x, x*0, label = "Modello")
plt.title("Residui battimenti p2")
plt.show()


# Battimenti prostaferesi
x = np.linspace(battimenti[0,0]-5, battimenti[0,-1]+5, 4000)
plt.figure("Battimenti prost")
plt.errorbar(battimenti[0], battimenti[1], yerr = 1, label = "Punti p1", fmt = 'o', color = "blue", markersize=5)
plt.plot(x, fit_prostaferesi(x, *popt_prost1), label = "Modello p1", color = "cyan")
plt.errorbar(battimenti[2], battimenti[3], yerr = 1, label = "Punti p2", fmt = 'o', color = "red", markersize=5)
plt.plot(x, fit_prostaferesi(x, *popt_prost2), label = "Modello p2", color = "orange")
plt.xlabel("[s]")
plt.ylabel("[ua]")
plt.title("Grafico battimenti prostaferesi")
plt.legend()
plt.show()

# Residui
plt.figure("R. prost")
plt.subplot(211)
plt.errorbar(battimenti[0], battimenti[1]-fit_prostaferesi(battimenti[0], *popt_prost1), yerr = 1, fmt = 'o')
plt.plot(x, x*0, label = "Modello")
plt.title("Residui battimenti prostaferesi p1")
plt.subplot(212)
plt.errorbar(battimenti[2], battimenti[3]-fit_prostaferesi(battimenti[2], *popt_prost2), yerr = 1, fmt = 'o')
plt.plot(x, x*0, label = "Modello")
plt.title("Residui battimenti prostaferesi p2")
plt.show()