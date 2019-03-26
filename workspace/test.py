import numpy as np
import matplotlib.pyplot as plt
import solvers
import linoplib


# def gaussian(x, offset, amp, std):
#     return offset*(1+amp*np.exp((-(x-x[x.shape[0]//2])**2)/(2*std**2)))

def gaussian(x, offset, amp, std):
    return offset+amp*np.exp((-(x-x[x.shape[0]//2])**2)/(2*std**2))


def F(_f):
    f = _f.copy()
    f[1] -= f[0]
    f[-1] -= f[-2]
    return f[1:-1]


N = 1023
x = np.linspace(0, 50e-9, N)
A = linoplib.laplacian_LDO(N)
# v0 = np.sin(np.pi*x/x[-1])
# v0 = np.linspace(0, 1, N)
v0 = np.zeros_like(x)
f = gaussian(x, 0, 1, 5e-9)

v = v0.copy()
# v[1:-1] = solvers.FMG(A, v0[1:-1], F(f), 0.667, 1, 10, 20)
v[1:-1] = solvers.weighted_jacobi(A, v0[1:-1], F(f), 0.667, 1000)

# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].set_ylabel('f')
# ax[0].set_xlabel('x')
# ax[0].plot(x[1:-1], A@v[1:-1], '-r', linewidth=4, alpha=0.5)
# ax[0].plot(x, f, '-k')
# ax[1].set_ylabel('v')
# ax[0].set_xlabel('x')
# ax[1].plot(x, v, '-r', linewidth=4, alpha=0.5)
# ax[1].plot(x, v0, '-k')

# plt.show()
