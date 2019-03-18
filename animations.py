import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import visual_solvers
import linoplib


def gaussian(x, offset, amp, std):
    return offset+amp*np.exp((-(x-x[x.shape[0]//2])**2)/(2*std**2))


def F(_f):
    f = -1*_f.copy()
    f[1] -= f[0]
    f[-1] -= f[-2]
    return f[1:-1]

n_iter = 10000
N = 129
x = np.linspace(0, 50e-9, N)
A = linoplib.laplacian_LDO(N)
v = np.zeros((n_iter, x.shape[0]))
# v[0] = np.sin(10*np.pi*x/x[-1])
f = gaussian(x, 0, 0.5e-2, 5e-9)

v[:, 1:-1] = visual_solvers.weighted_jacobi(A, v[:, 1:-1], F(f), 0.667, n_iter)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))


font_size = 16

ax[0].set_xlabel('$x$', size=font_size)
ax[0].set_ylabel('$f$', size=font_size)
ax[1].set_xlabel('$x$', size=font_size)
ax[1].set_ylabel('$v$', size=font_size)

ax[0].plot(x, f, '-b')

ax[0].text(0.1, 0.4, '$f$', transform=ax[0].transAxes, color='b', fontsize=font_size)
ax[0].text(0.1, 0.1, '$Av$', transform=ax[0].transAxes, fontsize=font_size)

iter_text = ax[0].text(0.18, 0.90, '', transform=ax[0].transAxes, fontsize=font_size)
line_v, = ax[1].plot(x, v[0], '-k')
line_f, = ax[0].plot(x[1:-1], A@v[0, 1:-1], '-k')

ax[0].set_ylim(-2.5e-3, 6e-3)
ax[1].set_ylim(-4, 1.5)


def init():
    line_v.set_ydata([np.nan]*x.shape[0])
    iter_text.set_text('')
    line_f.set_ydata([np.nan]*x[1:-1].shape[0])
    return line_v, iter_text, line_f


def animate(i):
    line_v.set_ydata(v[i])
    iter_text.set_text(f'Iteration: {i}')
    line_f.set_ydata(A@v[i, 1:-1])
    return line_v, iter_text, line_f

ani = animation.FuncAnimation(fig,
                              func=animate,
                              frames=n_iter,
                              init_func=init,
                              interval=1,
                              blit=True,
                              save_count=n_iter)

ani.save('jacobi_zeros.mp4', fps=60, dpi=200, extra_args=['-vcodec', 'libx264'])
# plt.show()
