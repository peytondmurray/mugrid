import numpy as np
import matplotlib.pyplot as plt
import solvers
import linoplib


# def gaussian(x, offset, amp, std):
#     return offset*(1+amp*np.exp((-(x-x[x.shape[0]//2])**2)/(2*std**2)))

def gaussian(x, offset, amp, std):
    return offset+amp*np.exp((-(x-x[x.shape[0]//2])**2)/(2*std**2))


def F(_f):
    f = -1*_f.copy()
    f[1] -= f[0]
    f[-1] -= f[-2]
    return f[1:-1]


N = 129
x = np.linspace(0, 50e-9, N)
A = linoplib.laplacian_LDO(N)
v0 = np.sin(np.pi*x/x[-1])
# v0 = np.linspace(0, 1, N)
f = gaussian(x, 0, 1, 5e-9)

v = v0.copy()
# v[1:-1] = solvers.FMG(A, v0[1:-1], F(f), 0.667, 1, 10, 20)
# v[1:-1] = solvers.weighted_jacobi(A, v0[1:-1], F(f), 0.667, 10)


I_h2h = linoplib.full_weighting(N)

x_2h = I_h2h@x
A_2h = linoplib.laplacian_LDO(int((N-1)/2))
v0_2h = I_h2h@v0
f_2h = I_h2h@f

v_2h = solvers.weighted_jacobi(A_2h, v0_2h[1:-1], f_2h[1:-1], 0.667, 100)
v = solvers.weighted_jacobi(A, v0[1:-1], f[1:-1], 0.667, 100)

font_size = 16

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
ax.set_ylabel('$f$', size=font_size)
ax.set_xlabel('$x$', size=font_size)
ax.plot(x, f, '-k', label='f')
ax.plot(x_2h[1:-1], A_2h@v_2h, '-r', label='$Av$ ($2h$ grid)')
ax.plot(x[1:-1], A@v, '-b', label='$Av$ ($h$ grid)')
ax.legend(fontsize=font_size)
ax.text(0.1, 0.8, '100 Jacobi Iterations', transform=ax.transAxes, fontsize=font_size)

plt.show()
# plt.savefig('coarse_grid_jacobi.svg', bbox_inches='tight')
