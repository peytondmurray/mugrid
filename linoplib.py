import numpy as np
import numba as nb


@nb.jit(nopython=True)
def laplacian_LDO(N):
    """Generate the laplacian linear differential operator for a grid with N points.
    """

    A = np.zeros((N-2, N-2))
    for i in range(1, N-3):
        A[i, i-1] = 1
        A[i, i] = -2
        A[i, i+1] = 1

    A[0, 0] = -2
    A[0, 1] = 1
    A[N-3, N-3] = -2
    A[N-3, N-4] = 1

    return A


@nb.jit(nopython=True)
def prolongation(N):
    """Generates the interpolation operator for going from a grid with (N-1)/2 points to a grid with N points:

        0.5     0       0
        1       0       0
        0.5     0.5     0   .   .   .
        0       1       0
        0       0.5     0.5
                .
                .
                .

    """

    I = np.zeros((N, int((N-1)/2)))
    I[0, 0] = 0.5
    I[-1, -1] = 0.5
    for i in range(1, N-1):
        if i % 2 == 0:
            I[i, int(i/2)] = 0.5
            I[i, int(i/2-1)] = 0.5
        else:
            I[i, int((i-1)/2)] = 1

    return I


@nb.jit(nopython=True)
def full_weighting(N):
    """Returns the restriction operator for going from a grid with 2N points to a grid with N points. This is just the
    transpose of the prolongation operator (up to a factor).
    """
    return np.transpose(prolongation(N))*0.5


def get_good_grid_sizes(N):
    a = np.zeros(N+1, dtype=int)
    a[0] = 7
    for i in range(1, N+1):
        a[i] = int(2*a[i-1] + 1)
    return a
