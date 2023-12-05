# import matplotlib.pyplot as plt
from pylab import *
import numpy as np


tf = 150.0
dt = 0.05
n = 76

f = open("xs.txt", 'r')

xs = np.zeros((int(tf/dt), n))


n_index = 0
t_index = 0
for line in f:
    if line == "-\n":
        n_index = n_index + 1
        t_index = 0
    else:
        xs[t_index, n_index] = float(line)
        t_index = t_index + 1

t = np.arange(0, int(tf/dt), 1)

plot(t[::5] * dt , xs[::5, :], 'k', alpha=0.3)
grid(True, axis='x')
xlim([0, t[-1] * dt])
show()

f.close()
