import matplotlib.pyplot as plt
import numpy as np
import math

x = np.linspace(-10,10,100)
z = 1/(1 + np.exp(-x))

plt.plot(x,z)
plt.xlabel("x")
plt.ylabel("Sigmoid(X)")

plt.show()

