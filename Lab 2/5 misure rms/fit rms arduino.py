import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def sinusoide(t, A, w, phi, B):
    return A*np.sin(w*t + phi) + B
    
def derivata(t, A, w, phi, B):
    return w*A*np.cos(w*t + phi)
    
t,V = np.genfromtxt('data.txt', skip_header =0, unpack = True)
dt = np.ones(256)*4
dV = np.ones(256)
dV_eff = dV
val = (225, 258*2*np.pi*10**-6, np.pi/2, 275)
for i in range(10):
    popt, pcov = scipy.optimize.curve_fit(sinusoide, t, V, val, dV_eff, absolute_sigma = False)
    chi_2 = np.sum(((V - sinusoide(t,*popt))/dV_eff)**2)
    print(chi_2)
    dV_eff = np.sqrt(((derivata(t,*popt))*dt)**2 + dV**2)
 
w = popt[1]
print('\nil chi quadro è', chi_2)
print('il chi quadro ridotto è', chi_2/252)
print('A =',popt[0], '+-', pcov[0][0]**0.5)
print('w =',popt[1], '+-', pcov[1][1]**0.5)
print('phi =',popt[2], '+-', pcov[2][2]**0.5)
print('B =',popt[3], '+-', pcov[3][3]**0.5)
print('f =', w/(2*np.pi))

plt.figure()
plt.subplot(211)
plt.title("onda")
x = np.linspace(0,256*220,4000)
plt.errorbar(t,V, dV, dt, fmt = '.', label='data')
plt.plot(x, sinusoide(x, *val), label='fit parametri iniziali')
plt.plot(x, sinusoide(x,*popt), label='fit parametri ottimali')
plt.xlabel('time [us]')
plt.ylabel('digit')
plt.legend()

plt.subplot(212)
plt.title("residui normalizzati")
plt.errorbar(t,(V-sinusoide(t,*popt))/dV_eff, fmt = '.', label='data')
plt.plot(x, x*0)
plt.xlabel('time [us]')
plt.ylabel('digit')
plt.legend()
plt.show()