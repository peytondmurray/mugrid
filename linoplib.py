import numpy as np


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

    if int(N) != N:
        raise ValueError(f'int({int(N)}) != {N}: N must be integer')
    elif int(N) % 2 == 0:
        raise ValueError('Grid with even number of points cannot be interpolated.')
    else:
        N = int(N)

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


def full_weighting(N):
    """Returns the restriction operator for going from a grid with 2N points to a grid with N points. This is just the
    transpose of the prolongation operator (up to a factor).
    """
    return np.transpose(prolongation(N))*0.5
