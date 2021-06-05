import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize, scipy.stats
from scipy.odr import odrpack

def linear(pars, x):
        return x*pars[0] + pars[1]

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
model = odrpack.Model(linear)
data = odrpack.RealData(xc, yc, sx=xc_err, sy=yc_err)
odr = odrpack.ODR(data, model, beta0=[-1., 1.])
out = odr.run()
popt_c, cov_c = out.beta, out.cov_beta
m_c, f_c = popt_c
dm_c , df_c = np.sqrt(cov_c.diagonal())
chi2_c = out.sum_square
pvalue_c = 2 - scipy.stats.chi2.cdf(np.sum(chi2_c), 2)
ndof = len(xc) - 2
print("\nchi2 = {}\nchi2/ndof = {}\np-value = {}".format(chi2_c, chi2_c/ndof, pvalue_c))
print("\nm =", m_c, "+-", dm_c, "\nf =", f_c**(-1), "+-", df_c/(f_c**2))

#divergente
print("\nLENTE DIVERGENTE\n")
model = odrpack.Model(linear)
data = odrpack.RealData(xd, yd, sx=xd_err, sy=yd_err)
odr = odrpack.ODR(data, model, beta0=[1., 1.])
out = odr.run()
popt_d, cov_d = out.beta, out.cov_beta
m_d, f_d = popt_d
dm_d , df_d = np.sqrt(cov_c.diagonal())
chi2_d = out.sum_square
pvalue_d = 2 - scipy.stats.chi2.cdf(np.sum(chi2_d), 2)
ndof = len(xd) - 2
print("\nchi2 = {}\nchi2/ndof = {}\np-value = {}".format(chi2_d, chi2_d/ndof, pvalue_d))
print("\nm =", m_d, "+-", dm_d, "\nf =", f_d**(-1), "+-", df_d/(f_d**2))

#grafici
#convergente
plt.figure("legge delle lenti sottili - lente convergente")
plt.plot(xc, linear(popt_c, xc), label = "best fit")
plt.errorbar(xc, yc, xerr=xc_err, yerr=yc_err, label ="punti sperimentali", fmt ='+')
plt.xlabel("1/p[cm]")
plt.ylabel("1/q[cm]")
plt.title("legge delle lenti sottili - lente convergente")
plt.legend()
plt.show()
#residui
plt.figure("residui conv")
plt.errorbar(xc, yc - linear(popt_c, xc), xc_err, yc_err, label = "punti sperimentali", fmt = "o")
plt.plot(xc, xc*0, label="Modello")
plt.xlabel("1/p[cm]")
plt.ylabel("1/q - modello[cm]")
plt.title("grafico dei residui (convergente)")
plt.legend()
plt.show()
#divergente
plt.figure("legge delle lenti sottili - lente divergente")
plt.plot(xd, linear(popt_d, xd), label = "best fit")
plt.errorbar(xd, yd, xerr=xd_err, yerr=yd_err, label ="punti sperimentali", fmt ='+')
plt.xlabel("-1/p[cm]")
plt.ylabel("1/q[cm]")
plt.title("legge delle lenti sottili - lente divergente")
plt.legend()
plt.show()
#residui
plt.figure("residui div")
plt.errorbar(xd, yd - linear(popt_d, xd), xd_err, yd_err, label = "punti sperimentali", fmt = "o")
plt.plot(xd, xd*0, label="Modello")
plt.xlabel("-1/p[cm]")
plt.ylabel("1/q - modello[cm]")
plt.title("grafico dei residui (divergente)")
plt.legend()
plt.show()





