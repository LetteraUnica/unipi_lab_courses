import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def funzione(V0, nV0, nV1, l, c) :
    return nV0*np.log(((np.exp(V0/nV1)-1)/l)+1)+c


data = np.genfromtxt("DATI.txt", unpack = True)

ddp_in = data[0][6:]
ddp_out = data[1][6:]/1000
ddp_out_errore = np.ones(len(ddp_out))/100  #Da cambiare
indices = [1, 4, 9, 13]
x = np.take(ddp_out, indices)
y = np.take(ddp_in, indices)
p0 = (1.5, 5, 1, 1.3)
popt = p0
popt, pcov = scipy.optimize.curve_fit(funzione, ddp_out, ddp_in, p0)
print(popt)

t = np.linspace(0, 0.4, 4000)
plt.figure()
plt.subplot(211)
plt.errorbar(ddp_out, ddp_in, ddp_out_errore, fmt='o')
plt.plot(t, funzione(t, *popt))
plt.subplot(212)
plt.errorbar(ddp_out, ddp_in-funzione(ddp_out, *popt), ddp_out_errore, fmt='o')
plt.plot(t, t*0)
plt.show()
