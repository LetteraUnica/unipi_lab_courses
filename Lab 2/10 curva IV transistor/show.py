import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.interpolate


data = np.genfromtxt("data.txt", unpack=True, skip_header=1)
data[0] *= 0.2

plt.figure()
plt.plot(data[0], data[1], 'o')
plt.show()