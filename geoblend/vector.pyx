
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

    :param source:
        The source image that will have its gradient conserved.
    :param reference:
        The reference image that will be used to sample for the boundary
        conditions.
    :param mask:
        ndarray where nonzero values represent the region
        of valid pixels in an image. The mask should be
        typed to uint8.

    .. todo:: source and reference may be uint16, but arthimetic operations need typecasting.
    """

    cdef int height = mask.shape[0]
    cdef int width = mask.shape[1]

    cdef unsigned int i, j, nj, ni, sj, si, ej, ei, wj, wi, neighbors
    cdef unsigned int idx = 0
    cdef double coeff, s

    cdef int n = np.count_nonzero(mask)
    cdef double[:] vector = np.empty(n, dtype=np.float64)

    assert source.shape[0] == height
    assert source.shape[1] == width
    assert reference.shape[0] == height
    assert reference.shape[1] == width

    for j in range(height):
        for i in range(width):

            if mask[j, i] == 0:
                continue

            neighbors = 0

            # Define indices of 4-connected neighbors
            nj = <unsigned int>(j - 1)
            ni = <unsigned int>(i)

            sj = <unsigned int>(j + 1)
            si = <unsigned int>(i)

            ej = <unsigned int>(j)
            ei = <unsigned int>(i + 1)

            wj = <unsigned int>(j)
            wi = <unsigned int>(i - 1)

            # Keep a running variable that represents the
            # element of the vector. This will be assigned
            # to the array at the end.
            coeff = 0.0
            s = source[j, i]

            if mask[nj, ni] == 0:
                coeff += 2 * reference[nj, ni]
                coeff -= 2 * source[nj, ni]
            else:
                neighbors += 1
                coeff -= 4 * source[nj, ni]

            if mask[sj, si] == 0:
                coeff += 2 * reference[sj, si]
                coeff -= 2 * source[sj, si]
            else:
                neighbors += 1
                coeff -= 4 * source[sj, si]

            if mask[ej, ei] == 0:
                coeff += 2 * reference[ej, ei]
                coeff -= 2 * source[ej, ei]
            else:
                neighbors += 1
                coeff -= 4 * source[ej, ei]

            if mask[wj, wi] == 0:
                coeff += 2 * reference[wj, wi]
                coeff -= 2 * source[wj, wi]
            else:
                neighbors += 1
                coeff -= 4 * source[wj, wi]

            coeff += (2 * neighbors + 8) * s

            # Assign the value to the output vector
            vector[idx] = coeff
            idx += 1

    return np.asarray(vector)
