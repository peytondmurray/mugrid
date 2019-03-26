import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import solvers
import linoplib
import tqdm


@nb.jit(nopython=True)
def gaussian(x, offset, amp, std):
    return offset+amp*np.exp((-(x-x[x.shape[0]//2])**2)/(2*std**2))


@nb.jit(nopython=True)
def F(_f):
    f = -1*_f
    f[1] -= f[0]
    f[-1] -= f[-2]
    return f[1:-1]


# @nb.jit(nopython=True, parallel=True)
def get_sum_sq_error(grid_sizes):

    sse = np.zeros_like(grid_sizes)

    for i in tqdm.trange(grid_sizes.shape[0]):
        N = grid_sizes[i]
        # print(N)

        x = np.linspace(0, 50e-9, N)
        A = -linoplib.laplacian_LDO(N)
        v0 = np.zeros_like(x)
        f = gaussian(x, 0, 1, 5e-9)

        v = solvers.weighted_jacobi(A, v0[1:-1], F(f), 0.667, 100)

        # ax.plot(x[1:-1], A@v, '-b', label=f'$Av$ (${gridlabel}$ grid)')
        # ax.plot(N, np.sum((f[1:-1]-A@v)**2), 'ob')

        sse[i] = np.sum((f[1:-1]-A@v)**2)

    return sse


if __name__ == "__main__":

    n_grids = 9

    grid_labels = [''] + [2**n for n in range(1, n_grids)]
    grid_sizes = linoplib.get_good_grid_sizes(n_grids)
    error = get_sum_sq_error(grid_sizes)

    # Original grid
    x = np.linspace(0, 50e-9, grid_sizes[-1])
    f = gaussian(x, 0, 1, 5e-9)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid_sizes, error, '-ok')
    font_size = 16
    ax.set_ylabel('$\Sigma (f-Av)^2$', size=font_size)
    ax.set_xlabel('$N$', size=font_size)

    plt.show()
    # plt.savefig('error_vs_gridsize.svg', bbox_inches='tight')
