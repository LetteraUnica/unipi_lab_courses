import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("quadra_integrata.txt", unpack = True)

plt.figure()
plt.errorbar(data[0], data[1], fmt = 'o')
plt.show()
