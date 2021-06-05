import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.odr import odrpack


def fit_smorzata(x, v0, damp):
    return v0*np.e**(-damp*x)

def odr_periodo_theta(B, theta) :
    return 2*np.pi*np.sqrt(l/980.7) * (1 + B[0]*theta**2 + B[1]*theta**4)
    
def fit_periodo_theta(theta, a, b) :
    return 2*np.pi*np.sqrt(l/980.7) * (1 + a*theta**2 + b*theta**4)


# Lettura dati
spessore = 2
ds = 0.05
l = 111.02
dl = 0.1
d = l+3
dd = 0.1

smorzata = np.genfromtxt("C:/Users/Lorenzo/Desktop/Lab fisica/10) Pendolo quadrifilare/Dati/OscillazioneSmorzata.txt", skip_header = 4, unpack = True)
normaleA = np.genfromtxt("C:/Users/Lorenzo/Desktop/Lab fisica/10) Pendolo quadrifilare/Dati/Oscillazione10.txt", skip_header = 4, unpack = True)

# Analisi smorzata
smorzata0 = smorzata[2][::2]
smorzata1 = smorzata[2][1::2]
t_smorzata = (smorzata0 + smorzata1) * 0.5
dt_smorzata = smorzata1 - smorzata0

v_smorzata = spessore*l/((dt_smorzata)*d)
errv_smorzata = ds*l/(dt_smorzata*d) + dl*spessore/(dt_smorzata*d) + dd*spessore*l/(dt_smorzata*d**2) + (8*10**-6)*spessore*l/(dt_smorzata*d)

periodo_smorzata = np.array([])
for i in range(len(t_smorzata)-2):
    periodo_smorzata = np.append(periodo_smorzata, t_smorzata[i+2] - t_smorzata[i])

# Fit smorzata
popt_smorzata, pcov_smorzata = scipy.optimize.curve_fit(fit_smorzata, t_smorzata, v_smorzata, sigma = errv_smorzata, absolute_sigma = True)
chi2_smorzata = ((v_smorzata-fit_smorzata(t_smorzata, *popt_smorzata)) / (errv_smorzata))**2
print("Parametri ottimali: {}\n".format(popt_smorzata))
print("Errori parametri: {}\n".format(np.sqrt(np.diagonal(pcov_smorzata))))
print("Chi2: {}, aspettato {}\n".format(chi2_smorzata.sum(), len(v_smorzata)))

# Analisi periodo - theta
normaleA0 = normaleA[2][::2]
normaleA1 = normaleA[2][1::2]
t_normaleA = (normaleA0 + normaleA1) * 0.5
dt_normaleA = normaleA1 - normaleA0

v_normaleA = spessore*l/(dt_normaleA*d)
errv_normaleA = ds*l/(dt_normaleA*d) + dl*spessore/(dt_normaleA*d) + dd*spessore*l/(dt_normaleA*d**2) + (8*10**-6)*spessore*l/(dt_normaleA*d)

periodoA = np.array([])
for i in range(len(t_normaleA)-2):
    periodoA = np.append(periodoA, t_normaleA[i+2] - t_normaleA[i])

thetaA = np.arccos(1-v_normaleA**2/(2*980.7*l))
err_thetaA = ((v_normaleA / (980.7*l)) * (errv_normaleA + (dl*v_normaleA) / (2*l))) / (np.sqrt(1-thetaA**2))

index = [len(thetaA)-1, len(thetaA)-2]
thetaA = np.delete(thetaA, index)
err_thetaA = np.delete(err_thetaA, index)

# Fit periodo - theta
model = odrpack.Model(odr_periodo_theta)
data = odrpack.RealData(thetaA, periodoA, sx=err_thetaA, sy=8*10**-6)
odr = odrpack.ODR(data, model, beta0=(1/16, 11/3072))
out = odr.run()
popt_normaleA, pcov_normaleA = out.beta, out.cov_beta
chi2_normaleA = out.sum_square
print("Parametri ottimali: {}, aspettati: {} {}\n".format(popt_normaleA, 1/16, round(11/3072, 5)))
print("Errori parametri: {}\n".format(np.sqrt(np.diagonal(pcov_normaleA))))
print("Chi2: {}, aspettato {}\n".format(chi2_normaleA, len(thetaA)))


# Grafico smorzata
t = np.linspace(0, 650, 4000)
plt.figure()
plt.subplot(211)
plt.errorbar(t_smorzata[::8], v_smorzata[::8], errv_smorzata[::8], fmt = 'o', label = 'Punti')
plt.plot(t, fit_smorzata(t, *popt_smorzata), color = 'red', label = 'Modello')
plt.legend()
plt.subplot(212)
plt.errorbar(t_smorzata[::8], v_smorzata[::8]-fit_smorzata(t_smorzata[::8], *popt_smorzata), errv_smorzata[::8], fmt = 'o')
plt.plot(t_smorzata, t_smorzata*0, color = 'red')
plt.xlabel("t [s]")
plt.ylabel("v [cm/s]")
plt.savefig("Oscillazione smorzata.png", bbox_inches='tight')
plt.show()

plt.figure()
index = [len(t_smorzata)-1, len(t_smorzata)-2]
t_smorzata = np.delete(t_smorzata, index)
plt.errorbar(t_smorzata, periodo_smorzata, fmt = 'o', label = "Punti")
plt.legend()
plt.xlabel("t [s]")
plt.ylabel("T [s]")
plt.savefig("Periodo smorzato", bbox_inches='tight')
plt.show()

# Grafico periodo theta
theta = np.linspace(0.2, 0.6, 4000)
plt.figure()
plt.subplot(211)
plt.errorbar(thetaA[::4], periodoA[::4], xerr = err_thetaA[::4], fmt = 'o', label = 'Punti')
plt.plot(theta, fit_periodo_theta(theta, *popt_normaleA), color = 'red', label = 'Modello')
plt.legend(loc = 'lower right')
plt.subplot(212)
plt.errorbar(thetaA[::4], periodoA[::4]-fit_periodo_theta(thetaA[::4], *popt_normaleA), err_thetaA[::4]*0.092, fmt = 'o')
plt.plot(theta, theta*0, color = 'red')
plt.xlabel("theta [rad]")
plt.ylabel("T [s]")
plt.savefig("Periodo normale 10.png", bbox_inches='tight')
plt.show()