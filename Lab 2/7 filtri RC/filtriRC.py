import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def Att(f, ft, a):
    return a/((1+(f/ft)**2)**0.5)

def derivata(f, ft, a):
    return -a*(f/ft)*((1+(f/ft)**2)**(-1.5))

def Bode(f, ft):
    return 20*(np.log10(ft/f))
    
def derivataBode(f, ft):
    return np.abs(-20/(np.log(10)*f))

Vout, dVout, f, df = np.genfromtxt('data.txt', skip_header = 1, unpack = True)
Vi = 6.08
dVi = 0.19
A = Vout/Vi
dA = (1/Vi)*dVout + (Vout/(Vi**2))*dVi

# Fit normale
dA_eff = dA
popt = (174,1) # parametri iniziali
for i in range(5):
    popt, pcov = scipy.optimize.curve_fit(Att, f, A, popt, dA_eff, absolute_sigma = False)
    chi_2 = np.sum(((A - Att(f,*popt))/dA_eff)**2)
    print(chi_2)
    dA_eff = np.sqrt(((derivata(f,*popt))*df)**2 + dA**2)

ndof = len(Vout)-2 
print('\nil chi quadro è', chi_2)
print('il chi quadro ridotto è', chi_2/ndof)
print('ft (frequenza di taglio) =',popt[0], '+-', pcov[0][0]**0.5)
print('a =',popt[1], '+-', pcov[1][1]**0.5)
print("Cov normalizzata =", pcov[1][0]/(pcov[0][0]*pcov[1][1])**0.5, "\n")

# Fit Bode
mask = f>400
ABode = 20*np.log10(A[mask])
fBode = f[mask]
dABode = 20*(dA[mask]/A[mask])/np.log(10)
dA_effB = dABode
dfBode = df[mask]
for i in range(5):
    poptBode, pcovBode = scipy.optimize.curve_fit(Bode, fBode, ABode, (1), dA_effB, absolute_sigma = False)
    chi_2 = np.sum(((ABode - Bode(fBode,*poptBode))/dA_effB)**2)
    print(chi_2)
    dA_effB = np.sqrt(((derivataBode(fBode, *poptBode))*dfBode)**2 + dABode**2)

print("ft =",poptBode, "+-", np.diag(pcovBode)**0.5, "\n")


# Grafici
# Plot funzione
x = np.logspace(1, 5, 4000)
plt.figure()
plt.subplot(211)
plt.xscale('log')
plt.yscale('log')
plt.errorbar(f, A, yerr=dA, xerr=df, fmt='.', label="Misure")
plt.plot(x, Att(x, *popt), label="Fit")
plt.xlabel("f [Hz]")
plt.ylabel("A")
plt.legend()

# Plot residui normalizzati
plt.subplot(212)
plt.xscale('log')
plt.errorbar(f, (A -Att(f, *popt))/dA_eff, fmt='.', label="Res. norm.")
plt.plot(x, x*0)
plt.xlabel("f [Hz]")
plt.legend()
plt.show()

# Bode plot
xB = np.linspace(0, 8, 2000)
plt.figure()
plt.subplot(211)
plt.xscale('log')
plt.errorbar(f, 20*np.log10(A), yerr=20*(dA/A)/np.log(10), xerr=df, fmt='.', label="Misure")
plt.plot(x, Bode(x, *poptBode), label="Fit")
plt.xlabel("f [Hz]")
plt.ylabel("A")
plt.legend()

# Bode residui normalizzati
plt.subplot(212)
plt.xscale('log')
plt.errorbar(fBode, (ABode - Bode(fBode, *poptBode))/dA_effB, fmt='.', label="Res. norm.")
plt.plot(x, x*0)
plt.xlabel("f [Hz]")
plt.legend()
plt.show()

