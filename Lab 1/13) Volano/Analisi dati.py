import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.odr import odrpack

raggi = np.array([15.9*2, 3.515, 5.905, 1.515, 1.515, 2.040])*0.5
errore_raggi = np.array([0.2, 0.005, 0.005, 0.005, 0.005, 0.005])*0.5
spessore = np.array([1.050, 1.5, 0.79, 2.18, 1.5, 3.815])
errore_spessore = np.array([0.005, 0.1, 0.05, 0.07, 0.1, 0.005])