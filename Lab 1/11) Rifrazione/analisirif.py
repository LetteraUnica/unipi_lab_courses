import numpy as np
import scipy.optimize, scipy.stats
import matplotlib.pyplot as plt
from scipy.odr import odrpack


def linear_odr(pars, x):
        return x*pars[0] + pars[1]

def linear(x, m, q):
        return x*m + q

# Lettura dati
rif_acqua = np.genfromtxt("/Users/Alicelongh/Documents/LAB/indice_di_rifrazione/acqua.txt", skip_header = 4, unpack = True)
rif_plexi = np.genfromtxt("/Users/Alicelongh/Documents/LAB/indice_di_rifrazione/plexi.txt", skip_header = 5, unpack = True)


# Analisi dati
x = np.linspace(0, 6, 1000)

rsin_i = rif_plexi[0]*0.2
rsin_r = rif_plexi[1]*0.2
err_sini = np.array(len(rsin_r)*[0.05])
err_sinr = np.array(len(rsin_r)*[0.05])
ndof_p = len(rsin_r) - 2
print(rsin_i, rsin_r)
print("\nPLEXIGLASS")

# Fit plexiglass
model = odrpack.Model(linear_odr)
data = odrpack.RealData(rsin_r, rsin_i, sx=err_sinr, sy=err_sini)
odr = odrpack.ODR(data, model, beta0=[1., 0.])
out = odr.run()
popt_p, cov_p = out.beta, out.cov_beta
nplexi, q = popt_p
dnplexi , dq = np.sqrt(cov_p.diagonal())
chi2 = out.sum_square
pvalue = 2 - scipy.stats.chi2.cdf(np.sum(chi2), 2)
print('chi2 = %.1f\nchi2/ndof = %.1f\np-value =%.1d' % (chi2, chi2/ndof_p, pvalue))
print('Indice di rifrazione plexigss: %.3f +- %.3f' % (nplexi, dnplexi))


#grafico plexi
plt.figure("legge di snell aria-plexiglass")
plt.subplot(211)
plt.plot(x, linear_odr(popt_p, x), label = "best fit")
plt.errorbar(rsin_r, rsin_i, err_sini, err_sinr,label ="punti sperimentali", fmt ='o')
plt.xlabel("R*sin(theta_r)[cm]")
plt.ylabel("R*sin(theta_i)[cm]")
plt.title("legge di Snell per il plexiglass")
plt.legend()

#residui
plt.subplot(212)
plt.errorbar(rsin_r, rsin_i - linear_odr(popt_p, rsin_r), err_sinr, err_sini, label = "punti sperimentali", fmt = "o")
plt.plot(x, x*0, label="Modello")
plt.xlabel("sin entrata - modello [cm]")
plt.ylabel("sin uscita[cm]")
plt.title("grafico dei residui")
plt.legend()
plt.show()


#dati acqua
p = 100.8 + 1.8 - rif_acqua[0]
q = rif_acqua[1] - 1.8
r = 6.6
k = 1/p
y = 1/q
err_p = np.array(len(p)*[0.5])
err_q = np.array(len(q)*[0.5])
err_y = err_q/(q**2)
err_k = err_p/(p**2)
err_r = 0.05
ndof_a = len(k) - 2
print(k, y, err_k, err_y)
print("\nACQUA")

#fit acqua
model = odrpack.Model(linear_odr)
data = odrpack.RealData(k, y, sx=err_k, sy=err_y)
odr = odrpack.ODR(data, model, beta0=[1., 0.])
out = odr.run()
popt_a, cov_a = out.beta, out.cov_beta
nacqua, b = popt_a
dnacqua , db = np.sqrt(cov_a.diagonal())
chi2 = out.sum_square
pvalue = 2 - scipy.stats.chi2.cdf(np.sum(chi2), 2)
print("chi2 = %.1f\nchi2/ndof = %.1f\np-value = %.1f" % (chi2, chi2/ndof_a, pvalue))
print("Indice di rifrazione acqua: %.3f +- %.3f" % (-nacqua, dnacqua))

#grafico acqua
l = np.linspace(0.02, 0.032, 500)
plt.figure("indice di rifrazione acqua")
plt.subplot(211)
plt.plot(l, linear_odr(popt_a, l), label = "best fit")
plt.errorbar(k, y, err_k, err_y, label ="punti sperimentali", fmt ='+')
plt.xlabel("1/distanza sorgente-diottro [cm]")
plt.ylabel("1/distanza diottro-immagine [cm]")
plt.title("indice di rifrazione per l'acqua")
plt.legend()

#residui
plt.subplot(212)
plt.errorbar(k, y - linear_odr(popt_a, k), err_k, err_y, label = "punti sperimentali", fmt = "o")
plt.plot(l, l*0, label="Modello")
plt.xlabel("distanza sorgente-diottro - modello [cm]")
plt.ylabel("distanza diottro-immagine [cm]")
plt.title("grafico dei residui acqua")
plt.legend()
plt.show()