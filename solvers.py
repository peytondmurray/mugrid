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
    D_inv = np.diag(1/diag)
    L_plus_U = D-A

    for _ in range(nu_1):
        v = (1-w)*v + w*D_inv@(f + L_plus_U@v)

    return v


def VMG(A, v0, f, w, nu_1=10, nu_2=20, depth=-1):

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

    v = weighted_jacobi(A, v, f, w, nu_1=nu_1)
    return v


def FMG(A, v0, f, w, nu_0, nu_1, nu_2):

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
