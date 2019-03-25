import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
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

cmap = mplcm.get_cmap('viridis')
font_size = 16
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
ax.set_ylabel('$v$', size=font_size)
ax.set_xlabel('$x$', size=font_size)
ax.set_yticklabels([])
ax.set_xlim(0, 50e-9)
ax.set_xticklabels([])

N = 255
x = np.linspace(0, 50e-9, N)
v0 = np.sin(2*np.pi*x/x[-1])
ax.plot(x, v0, '-k')

nsteps = 5
for i in range(nsteps):

    I_h2h = linoplib.full_weighting(x.shape[0])
    x = I_h2h@x
    v0 = I_h2h@v0

    ax.plot(x, v0 - 0.5*i, '-o', color=cmap(0.6*i/nsteps))


plt.show()
# plt.savefig('coarse_grid_jacobi.svg', bbox_inches='tight')
