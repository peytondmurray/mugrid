import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import solvers
import linoplib
import time


def gaussian(x, offset, amp, std):
    return offset+amp*np.exp((-(x-x[x.shape[0]//2])**2)/(2*std**2))


def F(_f):
    f = -1*_f.copy()
    f[1] -= f[0]
    f[-1] -= f[-2]
    return f[1:-1]


# @profile
def main():
    font_size = 16
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6), sharex=True, sharey=True)
    grid_sizes = linoplib.get_good_grid_sizes(8)

    err_jac = []
    err_vmg = []
    err_fmg = []

    for N in grid_sizes:
        print(N)
        x = np.linspace(0, 50e-9, N)
        A = linoplib.laplacian_LDO(N)
        v0 = np.zeros_like(x)
        f = gaussian(x, 0, 1, 5e-9)

        w = 0.667

        t0 = time.time()
        # v_jac = solvers.weighted_jacobi(A, v0[1:-1], f[1:-1], w, 1000)
        t1 = time.time()
        v_vmg = solvers.VMG(A, v0[1:-1], f[1:-1], w, nu_1=380, nu_2=380)
        t2 = time.time()
        # v_fmg = solvers.FMG(A, v0[1:-1], f[1:-1], w, nu_0=2, nu_1=400, nu_2=15)
        t3 = time.time()

        # err_jac.append(np.sum((f[1:-1] - A@v_jac)**2))
        # err_vmg.append(np.sum((f[1:-1] - A@v_vmg)**2))
        # err_fmg.append(np.sum((f[1:-1] - A@v_fmg)**2))

        # print(f'{N}:\t{t1-t0:+3f}\t{t2-t1:+3f}\t{t3-t2:+3f}')

    # ax.plot(grid_sizes, err_jac, '-ok', label='Jacobi')
    # ax.plot(grid_sizes, err_vmg, '-or', label='VMG')
    # ax.plot(grid_sizes, err_fmg, '-ob', label='FMG')

    # plt.show()
    # plt.savefig('coarse_grid_jacobi.svg', bbox_inches='tight')

if __name__ == '__main__':
    main()