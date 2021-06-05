import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import uncertainties as unc
from uncertainties import unumpy

R, dr, Iq, dIq, Vg, dVg, vd, dvd = np.genfromtxt("dati.txt", unpack=True, skip_header=1)
Iq = unumpy.uarray(Iq, dIq)*0.001
Vg = unumpy.uarray(Vg, dVg)
vd = unumpy.uarray(vd, dvd)*0.001
Ra = unc.ufloat(6.73e3, 0.06e3)
Rb = unc.ufloat(0.807e3, 0.07e3)
C = unc.ufloat(10e-6, 1e-6)
nVt = 52e-3

Rth = ((Ra*Rb)/(Ra+Rb))
Vth = (Rb/(Ra+Rb))*Vg
id = (Vth-vd)/Rth
rd_att = nVt/Iq
rd1 = vd/id
rd2 = (vd*Ra*Rb) / (Rb*Vg - vd*(Ra+Rb))
print(id)