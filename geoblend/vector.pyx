
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def create_vector(double[:, ::1] source, double[:, ::1] reference, char[:, ::1] mask):
    """
    Computes the column vector needed to solve the linearized Poisson equation.
    This vector returned preserves the gradient of the source image. Other functions
    may be written to preserve other vector fields.
    
    :param mask:
        ndarray where nonzero values represent the region
        of valid pixels in an image. The mask should be
        typed to uint8.
    :param source:
        The source image that will have its gradient conserved.
    :param reference:
        The reference image that will be used to sample for the boundary
        conditions.

    .. todo:: source and reference may be uint16, but arthimetic operations need typecasting.
    """

    cdef int height = mask.shape[0]
    cdef int width = mask.shape[1]
    
    cdef int i, j
    cdef unsigned int nj, ni, sj, si, ej, ei, wj, wi
    cdef unsigned int idx = 0
    cdef double coeff, s

    cdef int n = np.count_nonzero(mask)
    cdef double[:] vector = np.empty(n, dtype=np.float64)

    assert source.shape[0] == height
    assert source.shape[1] == width
    assert reference.shape[0] == height
    assert reference.shape[1] == width

    # TODO: nogil shiz?
    for j in range(height):
        for i in range(width):

            if mask[j, i] == 0:
                continue

            # Define indices of 4-connected neighbors
            nj, ni = j - 1, i
            sj, si = j + 1, i
            ej, ei = j, i + 1
            wj, wi = j, i - 1

            # Keep a running variable that represents the
            # element of the vector. This will be assigned
            # to the array at the end.
            coeff = 0.0
            s = source[j, i]

            coeff += 4 * (s - source[nj, ni])
            if mask[nj, ni] == 0:
                coeff += 2 * reference[nj, ni]

            coeff += 4 * (s - source[sj, si])
            if mask[sj, si] == 0:
                coeff += 2 * reference[sj, si]

            coeff += 4 * (s - source[ej, ei])
            if mask[ej, ei] == 0:
                coeff += 2 * reference[ej, ei]

            coeff += 4 * (s - source[wj, wi])
            if mask[wj, wi] == 0:
                coeff += 2 * reference[wj, wi]

            # Assign the value to the output vector
            vector[idx] = coeff
            idx += 1

    return np.asarray(vector)
