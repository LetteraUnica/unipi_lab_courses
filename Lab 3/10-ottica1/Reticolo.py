import pylab
from scipy.optimize import curve_fit
import numpy
import numpy

########################################################################### Funzioni ##########################################################################

def d( λ, thetai, thetad):
    return λ / (numpy.sin(thetai) - numpy.sin(thetad))
    
def dd(λ, thetai, thetad, dthetai, dthetad):
    return λ * (numpy.sin(thetai) - numpy.sin(thetad))**(-2) * (numpy.cos(thetai)*dthetai + numpy.cos(thetad)*dthetad)
    
    
    
def lambdaH1(d, thetai, thetad):
    return d * (numpy.sin(thetai) - numpy.sin(thetad))
    
def DlambdaH1(d, dd, thetai, thetad, dthetai, dthetad):
    return ((numpy.sin(thetai) - numpy.sin(thetad))* dd) + d*(numpy.cos(thetai)*dthetai + numpy.cos(thetad)*dthetad)
    
    
    
def lambdaH2(d, thetai, thetad):
    return (d/2) * (numpy.sin(thetai) - numpy.sin(thetad))

def DlambdaH2(d, dd, thetai, thetad, dthetai, dthetad):
    return ((numpy.sin(thetai) - numpy.sin(thetad)) * dd/2) + (d/2)*(numpy.cos(thetai)*dthetai + numpy.cos(thetad)*dthetad)
        
def Thetai(theta0):
    return 0.5 * (numpy.pi - theta0)
    
def Thetad(theta, theta0):
    return numpy.pi - theta - theta0
    
def lambdainv(n2, R):
    return R * (1/(4.) - 1/(n2**2))
    
def degrad(x):
    return x * numpy.pi / 180
    
############################################################################# Dati ###########################################################################

beta0 = degrad(168.565)
dbeta0 = degrad(1/120)

dtheta = 2 * dbeta0

Rth = 10973731.568

# Stima di "d"

theta0 = degrad(205.00) - beta0
theta1 = degrad(259.64) - beta0 



thetai = Thetai(theta0)

λ = 546.074*1e-9

thetad = Thetad(theta1, thetai)
dthetad = dtheta + dbeta0

d = d( λ, thetai, thetad)
dd = dd(λ, thetai, thetad, dtheta, dthetad)

print('\n__________________ Stima di "d" __________________\n')
print('d = %.7f +- %.7f [righe/mm]' % (1/d * 1e-3, (1/d**2)*dd * 1e-3))
print('\n__________________________________________________\n')


# Stima di "R"

theta0 = degrad(204.79) - beta0

theta1 = degrad(pylab.array([251.35, 251.45, 255.26, 258.74, 259.50, 264.62, 267.45])) - beta0

theta2 = degrad(pylab.array([282.08, 282.46, 289.34])) - beta0

thethai = Thetai(theta0)

theta1d = Thetad(theta1, thetai)

theta2d = Thetad(theta2, thetai)

λpo = lambdaH1(d, thetai, theta1d)
dλpo = DlambdaH1(d, dd, thetai, theta1d, dtheta, dthetad)

λso = lambdaH2(d, thetai, theta2d)
dλso = DlambdaH2(d, dd, thetai, theta2d, dtheta, dthetad)

λtot = numpy.concatenate((λpo, λso))
dλtot = numpy.concatenate((dλpo, dλso)) 

λ = pylab.array([λpo[0], λpo[2], λpo[6], λso[0], λso[2]])
dλ = pylab.array([dλpo[0], dλpo[2], dλpo[6], dλso[0], dλso[2]]) 


n1 = pylab.array([2., 2., 2., 2., 2.])##########################
n2 = pylab.array([5., 4., 3., 5., 4.])##############################
dn2 = n2 *0 + 0.

#fit
popt, pcov = curve_fit(lambdainv, n2, 1/λ, p0 = Rth , sigma = (1/λ**2)*dλ, absolute_sigma = True)
R      = popt
dR     = pylab.sqrt(pcov.diagonal())

print('\n__________________ Stima di "R" __________________\n')
print('λpo = {} +- {} [m]' .format(λpo, dλpo))
print('λso = {} +- {} [m]' .format(λso, dλso))
print('------------------> R = %.4f +- %.4f' % (R*1e-7, dR*1e-7))
print('\n__________________________________________________\n')


# Lunghezza d'onda del dopietto del sodio

theta1 = degrad(pylab.array([262.70, 262.73])) - beta0
theta2 = degrad(pylab.array([304.42, 304.51])) - beta0

thethai = Thetai(theta0)

theta1d = Thetad(theta1, thetai)
theta2d = Thetad(theta2, thetai)

λi = lambdaH1(d, thetai, theta1d)
dλi = DlambdaH1(d, dd, thetai, theta1d, dtheta, dthetad)





############################################################################ Grafici ##########################################################################
# X  =  n2 Y = 1/λ
x = numpy.linspace(2., 6., 1000)
y = lambdainv(x, *popt)

pylab.figure(1)
pylab.subplot(211)
pylab.title('Costante di Rydberg')
pylab.ylabel('1/$\lambda$ [1/m]')
pylab.xlabel('n2 [u.a.]')
pylab.grid(color = 'gray')
pylab.errorbar(n2, (1./λ), (1./λ**2)*dλ, dn2,  '.', color = 'Black', label="Dati" )

pylab.plot(x, y, color = 'r', label="Fit")
pylab.legend()

pylab.subplot(212)
pylab.plot(x, x*0, color="red", label="Fit")
pylab.plot(n2, ((1./λ)-lambdainv(n2, *popt))/((1./λ**2)*dλ), 'o', color="black", label="Dati")
pylab.legend()
pylab.savefig('lambdan2.pdf')

pylab.show()

print('\n__________________ Doppietto sodio __________________\n')
print('λ1 = %.10f +- %.10f [m]' % (λi[0], dλi[0]))
print('λ2 = %.10f +- %.10f [m]' % (λi[1], dλi[1]))
print('\n_____________________________________________________\n')

































