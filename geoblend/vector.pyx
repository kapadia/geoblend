
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def create_vector(unsigned short[:, ::1] source, unsigned short[:, ::1] reference, char[:, ::1] mask, double multiplier = 1.0):
    """
    Computes the column vector needed to solve the linearized Poisson equation.
    This vector returned preserves the gradient of the source image. Other functions
    may be written to preserve other vector fields.

    :param source:
        The source image represented by a uint16 ndarray. This
        image will have its gradient conserved in the blend process.
    :param reference:
        The reference image represented by a uint16 ndarray. This
        image will be used to sample for the boundary conditions.
    :param mask:
        ndarray where nonzero values represent the region
        of valid pixels in an image. The mask should be
        typed to uint8.
    :param multipler:
        Scaling factor to apply to the gradient. This is useful when working with images that
        live in different regions of the dynamic range.
    """

    cdef int height = mask.shape[0]
    cdef int width = mask.shape[1]

    cdef unsigned int i, j, nj, ni, sj, si, ej, ei, wj, wi, neighbors
    cdef unsigned int idx = 0
    cdef double coeff, s

    cdef int n = np.count_nonzero(mask)

    # PyAMG requires a double typed vector
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
            s = multiplier * source[j, i]

            if mask[nj, ni] == 0:
                coeff += <double>(2 * reference[nj, ni])
                coeff -= <double>(2 * multiplier * source[nj, ni])
            else:
                neighbors += 1
                coeff -= <double>(4 * multiplier * source[nj, ni])

            if mask[sj, si] == 0:
                coeff += <double>(2 * reference[sj, si])
                coeff -= <double>(2 * multiplier * source[sj, si])
            else:
                neighbors += 1
                coeff -= <double>(4 * multiplier * source[sj, si])

            if mask[ej, ei] == 0:
                coeff += <double>(2 * reference[ej, ei])
                coeff -= <double>(2 * multiplier * source[ej, ei])
            else:
                neighbors += 1
                coeff -= <double>(4 * multiplier * source[ej, ei])

            if mask[wj, wi] == 0:
                coeff += <double>(2 * reference[wj, wi])
                coeff -= <double>(2 * multiplier * source[wj, wi])
            else:
                neighbors += 1
                coeff -= <double>(4 * multiplier * source[wj, wi])

            coeff += <double>((2 * neighbors + 8) * s)

            # Assign the value to the output vector
            vector[idx] = coeff
            idx += 1

    return np.asarray(vector)


@cython.boundscheck(False)
@cython.wraparound(False)
def create_vector_from_field(double[:, ::1] source, unsigned short[:, ::1] reference, char[:, ::1] mask):
    """
    Computes the column vector needed to solve the linearized Poisson equation.

    :param source:
        The vector field that will be preserved.
    :param reference:
        The reference image represented by a uint16 ndarray. This
        image will be used to sample for the boundary conditions.
    :param mask:
        ndarray where nonzero values represent the region
        of valid pixels in an image. The mask should be
        typed to uint8.
    """

    cdef int height = mask.shape[0]
    cdef int width = mask.shape[1]

    cdef unsigned int i, j, nj, ni, sj, si, ej, ei, wj, wi, neighbors
    cdef unsigned int idx = 0
    cdef double coeff, s

    cdef int n = np.count_nonzero(mask)

    # PyAMG requires a double typed vector
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
                coeff += <double>(2 * reference[nj, ni])
            else:
                neighbors += 1

            if mask[sj, si] == 0:
                coeff += <double>(2 * reference[sj, si])
            else:
                neighbors += 1

            if mask[ej, ei] == 0:
                coeff += <double>(2 * reference[ej, ei])
            else:
                neighbors += 1

            if mask[wj, wi] == 0:
                coeff += <double>(2 * reference[wj, wi])
            else:
                neighbors += 1

            coeff += <double>(neighbors * s)

            # Assign the value to the output vector
            vector[idx] = coeff
            idx += 1

    return np.asarray(vector)
