import pylab
from scipy.optimize import curve_fit
import numpy


########################################################################### Funzioni #########################################################################

def f(x, a, b):
    return a*x + b
    
def degrad(x):
    return x * numpy.pi / 180
    
############################################################################# Dati ###########################################################################

beta0 = 302.87
dbeta0 = 1/60

δ = pylab.array([253.13, 253.37, 253.73, 255.16]) - beta0
dδ = δ * 0 + (1/60) 

δ = degrad(δ)
dδ = degrad(dδ)

λ = pylab.array([467.8, 480.0, 508.6, 643.8]) * 10e-9
dλ = λ * 0


δNa = 254.65 - beta0
dδNa = 1/60

δNa = degrad(δNa)
dδNa = degrad(dδNa)

############################################################################## Fit ###########################################################################
#fit
popt, pcov = curve_fit(f, 1./λ, δ, sigma = dδ, absolute_sigma = False)
a, b       = popt
da, db     = pylab.sqrt(pcov.diagonal())


λNa = a / (δNa - b)
dλNa = ( ((1 / (δNa - b))*da) + (a / ((δNa - b)**2))*(dδNa + db) )


#valori stimati
print('\n________________________________________________\n')
print('a= %.9f +- %.9f [m]' % (a, da))
print('b = %.3f +- %.3f [rad]' % (b,db))
print('--------------------------> λNa = %.8f +- %.8f [m]' % (λNa,dλNa))
print('\n________________________________________________\n')

############################################################################ Grafici ##########################################################################
# X  = 1/λ Y = δ

pylab.figure(1)
pylab.title('Indice di rifrazione dell\'acqua')
pylab.xlabel('1/$\lambda$ [1/m]')
pylab.ylabel('$\sigma$ [rad]')
pylab.grid(color = 'gray')
pylab.errorbar((1./λ), δ, dδ, dλ, 'o', color = 'Black' )
pylab.ylim(-0.87, -0.83)
pylab.xlim(150000., 220000.)

x = numpy.linspace(150000., 220000., 10000)
y = f(x, *popt)

pylab.plot(x, y, color = 'r')

#pylab.savefig('lambdadelta.pdf')

pylab.show()