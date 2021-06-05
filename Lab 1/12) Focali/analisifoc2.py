import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize, scipy.stats
from scipy.odr import odrpack

def linear(x, m, q):
        return x*m + q

#dati
#convergente
[pc, qc] = np.genfromtxt("/Users/Alicelongh/Documents/LAB/misure_focali/convergente.txt", skip_header = 2, unpack = True)
xc = pc**(-1)
yc = qc**(-1)
xc_err = 0.2/(pc**2)
print(xc_err)
#np.put(xc_err, [1, 5, 6], [+0.0007, +0.0007, +0.0007])
yc_err = 0.2/(qc**2)
np.put(yc_err, [1, 5, 6], [+0.0007, +0.0007, +0.0007])
np.put(yc_err, [0, 2, 3, 4, 7, 8, 9], [+0.0004, +0.0004, +0.0004, +0.0004, +0.0004, +0.0004, +0.0004])
print(yc_err)
#divergente
[pd, qd] = np.genfromtxt("/Users/Alicelongh/Documents/LAB/misure_focali/divergente.txt", skip_header = 2, unpack = True)
xd = pd**(-1)
yd = qd**(-1)
xd_err = 0.2/(pd**2)
yd_err = 0.2/(qd**2)

#fit
#convergente
print("\nLENTE CONVERGENTE\n")
[popt_c, cov_c] = scipy.optimize.curve_fit(linear, xc, yc, [1., 1.], sigma = yc_err)
der_yx = popt_c[0]
ndof = len(xc) - 2
for i in range(3):
        dxy_c = np.sqrt(yc_err**2 + (popt_c[0]*xc_err)**2)
        popt_c, cov_c = scipy.optimize.curve_fit(linear, xc, yc, popt_c, sigma = dxy_c)
        chi2 = ((yc - linear(xc, *popt_c))/dxy_c)**2.
        pvalue = 2 - scipy.stats.chi2.cdf(np.sum(chi2), 2)
        print("passo %d" % i)
        print("m = %.3f +- %.3f" % (popt_c[0], np.sqrt(cov_c.diagonal())[0]))
        print("f = %.3f +- %.3f" % (1/popt_c[1], np.sqrt(cov_c.diagonal())[1]/popt_c[1]**2))
        print("chi2 = %.3f, chi2/ndof = %.3f, pvalue = %.3f" % (chi2.sum(), chi2.sum()/ndof, pvalue))
#divergente
print("\nLENTE DIVERGENTE\n")
[popt_d, cov_d] = scipy.optimize.curve_fit(linear, xd, yd, [1., 1.], sigma = yd_err)
der_yx = popt_d[0]
ndof = len(xd) - 2
for i in range(5):
        dxy_d = np.sqrt(yd_err**2 + (popt_d[0]*xd_err)**2)
        popt_d, cov_d = scipy.optimize.curve_fit(linear, xd, yd, popt_d, sigma = dxy_d)
        chi2 = ((yd - linear(xd, *popt_d))/dxy_d)**2.
        pvalue = 2 - scipy.stats.chi2.cdf(np.sum(chi2), 2)
        print("passo %d" % i)
        print("m = %.3f +- %.3f" % (popt_d[0], np.sqrt(cov_d.diagonal())[0]))
        print("f = %.3f +- %.3f" % (1/popt_d[1], np.sqrt(cov_d.diagonal())[1]/popt_d[1]**2))
        print("chi2 = %.3f, chi2/ndof = %.3f, pvalue = %.3f" % (chi2.sum(), chi2.sum()/ndof, pvalue))
#grafici
#convergente
plt.figure("legge delle lenti sottili - lente convergente")
plt.plot(xc, linear(xc, *popt_c), label = "best fit")
plt.errorbar(xc, yc, xerr=xc_err, yerr=dxy_c, label ="punti sperimentali", fmt ='+')
plt.xlabel("1/p[cm]")
plt.ylabel("1/q[cm]")
#plt.title("legge delle lenti sottili - lente convergente")
plt.legend()
plt.show()
#residui
plt.figure("residui conv")
plt.errorbar(xc, yc - linear(xc, *popt_c), dxy_c, label = "punti sperimentali", fmt = "o")
plt.plot(xc, xc*0, label="Modello")
plt.xlabel("1/p[cm]")
plt.ylabel("1/q - modello[cm]")
#plt.title("grafico dei residui (convergente)")
plt.legend()
plt.show()
#divergente
plt.figure("legge delle lenti sottili - lente divergente")
plt.plot(xd, linear(xd, *popt_d), label = "best fit")
plt.errorbar(xd, yd, xerr=xd_err, yerr=dxy_d, label ="punti sperimentali", fmt ='+')
plt.xlabel("-1/p[cm]")
plt.ylabel("1/q[cm]")
#plt.title("legge delle lenti sottili - lente divergente")
plt.show()
#residui
plt.figure("residui div")
plt.errorbar(xd, yd - linear(xd, *popt_d), dxy_d, label = "punti sperimentali", fmt = "o")
plt.plot(xd, xd*0, label="Modello")
plt.xlabel("-1/p[cm]")
plt.ylabel("1/q - modello[cm]")
#plt.title("grafico dei residui (divergente)")
plt.show()












