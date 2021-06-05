import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import uncertainties as unc
from uncertainties import unumpy

Ib, dIb, Ic, dIc = np.genfromtxt("dati3.txt", unpack=True, skip_header=1)
Ib = unumpy.uarray(Ib, dIb)
Ic = unumpy.uarray(Ic, dIc)*1000

Bf = Ic/Ib
print(Bf)