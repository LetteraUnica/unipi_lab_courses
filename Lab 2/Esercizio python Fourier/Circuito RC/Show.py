import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("100hZ.txt", unpack = True)

plt.figure()
plt.errorbar(data[0], data[1])
plt.show()
