import pylab
from scipy.optimize import curve_fit
import numpy
import menzalib as mz


########################################################################### Funzioni #########################################################################

def f(x, a, b):
    return a*x + b
    
def degrad(x):
    return x * numpy.pi / 180

def int_retta(a, b, y0):
    return a/(y0-b)
    
############################################################################# Dati ###########################################################################

beta0 = 302.87
dbeta0 = 1/60

δ = pylab.array([253.13, 253.37, 253.73, 255.16]) - beta0
dδ = δ * 0 + (2/60)
δ = degrad(δ)
dδ = degrad(dδ)

λ = pylab.array([467.8, 480.0, 508.6, 643.8]) * 1e-9
dλ = λ * 0


δNa = 254.65 - beta0
dδNa = 2/60

δNa = degrad(δNa)
dδNa = degrad(dδNa)


############################################################################## Fit ###########################################################################
#fit
popt, pcov = curve_fit(f, 1./λ, δ, sigma = dδ, absolute_sigma = False)
a, b       = popt
da, db     = pylab.sqrt(pcov.diagonal())

cov = numpy.zeros((3,3))
cov[0:2, 0:2]=pcov
cov[2,2]=dδNa**2
x = numpy.hstack((popt, δNa))

λNa = int_retta(*x)*1e9
dλNa = numpy.sqrt(mz.dy(int_retta, x, cov))*1e9

#valori stimati
print('\n________________________________________________\n')
print('a= %.9f +- %.9f [m]' % (a, da))
print('b = %.3f +- %.3f [rad]' % (b,db))
print("chi2, pval: ", mz.chi2_pval(f, 1./λ, δ, dδ, popt))
print("λNa: {} +- {} [nm]".format(λNa, dλNa))
print('\n________________________________________________\n')

############################################################################ Grafici ##########################################################################
# X  = 1/λ Y = δ
x = numpy.linspace(1500000., 2200000., 1000)
y = f(x, *popt)
 
# Fit
pylab.figure(1)
pylab.subplot(211)
pylab.xlabel('1/$\lambda$ [1/m]')
pylab.ylabel('$\sigma$ [rad]')
pylab.grid(linestyle=':')
pylab.errorbar((1./λ), δ, dδ, dλ, 'o', color = 'black', label="Data")
pylab.plot(x, y, color = 'red', label="Fit")
pylab.plot(x, numpy.ones(1000)*δNa, linewidth=0.9)
pylab.legend()

# Residui
pylab.subplot(212)
pylab.grid(linestyle=':')
pylab.xlabel('1/$\lambda$ [1/m]')
pylab.ylabel('$\sigma$ [rad]')
pylab.plot(x,x*0, color="red", label="Fit")
pylab.plot(1/λ, (f(1/λ, *popt)-δ)/dδ, 'o', color="black", label="Residui")
pylab.legend()
#pylab.savefig('lambdadelta.pdf')

pylab.show()