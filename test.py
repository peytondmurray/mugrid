import numpy as np
import matplotlib.pyplot as plt
import solvers
import linoplib


def gaussian(x, offset, amp, std):
    return offset*(1+amp*np.exp((-(x-x[x.shape[0]//2])**2)/(2*std**2)))


N = 129
x = np.linspace(0, 50e-9, N)
A = linoplib.laplacian_LDO(N)
v0 = np.linspace(0, 1, N)
f = gaussian(x, 1e6, 10, 5e-9)

v = v0.copy()
v[1:-1] = solvers.FMG(A, v0[1:-1], f[1:-1], 0.667, 1, 10, 20)

fig, ax = plt.subplots()
ax.plot(x[1:-1], A@v[1:-1], '-r', linewidth=4, alpha=0.5)
ax.plot(x, f, '-k')

plt.show()
