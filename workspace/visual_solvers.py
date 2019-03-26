import numpy as np
import linoplib
import numba as nb


@nb.jit(nopython=True)
def weighted_jacobi(A, v, f, w, nu_1):
    """Weighted Jacobi solver. See linoplib.weighted_jacobi for more info on the input parameters. This version of the
    solver takes a 2D array as input for v; the first row is the initial guess, and each iteration is written to the
    subsequent rows of v (so that the progress of the solver can be visualized later).
    """

    diag = np.diag(A)
    D = np.diag(diag)
    D_inv = np.diag(1/diag)
    L_plus_U = D-A

    for i in range(1, nu_1):
        v[i] = (1-w)*v[i-1, :] + w*D_inv@(f + L_plus_U@v[i-1, :])

    return v