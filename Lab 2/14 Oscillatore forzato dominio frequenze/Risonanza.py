import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

R = 328

def A(f, r, C, f0) :
    return (2*np.pi*f*R*C)/(((2*np.pi*f*(R+r)*C)**2 + (1-(f/f0)**2)**2)**0.5)
    

f, Vout, dVout, Vin= np.genfromtxt('dati oscillatore forzato.txt', skip_header = 1, unpack = True)

dVin = np.sqrt(0.04**2 + (0.03*Vin)**2)
dVout = np.sqrt(dVout**2 + (0.03*Vout)**2)
datix = f
datiy = np.absolute(Vout/Vin)
dy = ((Vout*dVin+Vin*dVout)/(Vin)**2)

popt, pcov = scipy.optimize.curve_fit(A, datix, datiy, (50, 0.1e-6, 700), dy)
chi_2 = np.sum(((datiy - A(datix,*popt))/dy)**2)
ndof = len(f)-3
 
print('\nIl chi quadro è', chi_2)
print('il chi quadro ridotto è', chi_2/ndof)
print('r =',popt[0], '+-', pcov[0][0]**0.5)
print('C =',popt[1], '+-', pcov[1][1]**0.5)
print('f0 =',popt[2], '+-', pcov[2][2]**0.5)
print('Cov(r,C) =',pcov[0][1]/np.sqrt(pcov[0][0]*pcov[1][1]))
print('Cov(r,f0) =',pcov[0][2]/np.sqrt(pcov[0][0]*pcov[2][2]))
print('Cov(C,f0) =',pcov[1][2]/np.sqrt(pcov[1][1]*pcov[2][2]))


t = np.linspace(100, 1600, 4000)
plt.figure()
plt.subplot(211)
plt.grid()
plt.title("Curva di risonanza")
plt.ylabel("A(f)")
plt.errorbar(datix, datiy, dy, fmt='.', label = "Dati")
plt.plot(t, A(t, *popt), label = "Fit")
plt.legend()

plt.subplot(212)
plt.title("Residui normalizzati")
plt.xlabel("f [Hz]")
plt.errorbar(datix, (datiy - A(datix, *popt))/dy, fmt=".")
plt.plot(t, t*0)
plt.show()