
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

tempo, misure = np.genfromtxt("condensatore2tau_S.txt", skip_header=0, unpack=True)
dt = np.ediff1d(tempo)

print("tempo medio", np.average(dt), "+-", np.std(dt), "\n")
plt.figure()
plt.title("ddp [digit]")
plt.hist(misure, bins=3)
plt.show()

plt.figure()
plt.title("Tempo tra diverse misurazioni [us]")
plt.hist(dt, bins = 9)
plt.show()