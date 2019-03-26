import numpy as np
import numba as nb
import linoplib


def jacobi(A, v0, f, nu_1=1):
    return weighted_jacobi(A, v0, f, 1, nu_1)


@nb.jit(nopython=True)
def weighted_jacobi(A, v0, f, w, nu_1):

    v = v0.copy()
    diag = np.diag(A)
    D = np.diag(diag)
    Q = D-A

    for _ in range(nu_1):
        v = (1-w)*v + w*(f + Q@v)/diag

    return v


# @nb.jit(nopython=True)
def VMG(A, v0, f, w, nu_1=10, nu_2=10, depth=-1):
    """V-cycle multigrid solver. Solves Av = f.

    Parameters
    ----------
    A : np.ndarray
        Linear differential operator (Av = f).
    v0 : np.ndarray
        Initial guess for the solution to (Av = f)
    f : np.ndarray
        Forcing function (Av = f)
    w : float
        Jacobi weighting factor. Usually set to 2/3
    nu_1 : int, optional
        Number of jacobi iterations before each coarsening grid spacing step.
    nu_2 : int, optional
        Number of jacobi iterations at each de-coarsening grid spacing step.
    depth : int, optional
        Number of grid coarsening steps to take. Leave as -1 except for debugging.

    Returns
    -------
    np.ndarray
        1D array containing the solution.
    """

    print(f'\t{depth}')

    N = v0.shape[0]
    v = weighted_jacobi(A, v0, f, w, nu_1=nu_1)

    if depth != 0 and (N-1) % 2 == 0 and (N-1) > 8:

        I_h2h = linoplib.prolongation(N)
        I_2hh = linoplib.full_weighting(N)

        A_2h = I_2hh@A@I_h2h
        v_2h = np.zeros(int((N-1)/2))
        f_2h = I_2hh@(f-A@v)
        v_2h = VMG(A_2h, v_2h, f_2h, w, nu_1, nu_2, depth-1)
        v += I_h2h@v_2h

    v = weighted_jacobi(A, v, f, w, nu_1=nu_2)
    return v


@nb.jit(nopython=True)
def FMG(A, v0, f, w, nu_0=1, nu_1=10, nu_2=10):

    N = v0.shape[0]

    if (N-1) % 2 != 0 or (N-1) <= 8:
        v = np.zeros(N)
    else:
        I_h2h = linoplib.prolongation(N)
        I_2hh = linoplib.full_weighting(N)

        A_2h = I_2hh@A@I_h2h
        v_2h = np.zeros(int((N-1)/2))
        f_2h = I_2hh@(f-A@v0)
        v_2h = FMG(A_2h, v_2h, f_2h, w, nu_0, nu_1, nu_2)
        v = v0 + I_h2h@v_2h

    for _ in range(nu_0):
        v = VMG(A, v, f, w, nu_1, nu_2, depth=-1)

    return v


def direct(A, v0, f):
    return np.linalg.inv(A)@f


def error(A, v, f):
    return f-A@v
