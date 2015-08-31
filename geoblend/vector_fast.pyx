
import numpy as np
cimport numpy as np
cimport cython

DTYPE_FLOAT = np.float
ctypedef np.float_t DTYPE_FLOAT_t


@cython.boundscheck(False)
@cython.wraparound(False)
def create_vector(double[:, ::1] source, double[:, ::1] reference, char[:, ::1] mask):
    """
    Computes the column vector b from the linearized Poisson equation.
    
    :param mask:
        ndarray where nonzero values represent the region
        of valid pixels in an image. The mask should be
        typed to uint8.
    :param source:
        The source image that will have its gradient conserved.
    :param reference:
        The reference image that will be used to sample for the boundary
        conditions.
    
    .. todo:: The guidance field passed has thus far been an approximation of
              the gradient. A more precise representation may be calculated here
              if the source image is passed. Plus it will save the loop required
              when convolving the source image to find the gradient field.

    .. todo:: Can field be Int16?
    """

    cdef int height = mask.shape[0]
    cdef int width = mask.shape[1]

    assert source.shape[0] == height
    assert source.shape[1] == width
    assert reference.shape[0] == height
    assert reference.shape[1] == width

    cdef int nj, ni, ej, ei, sj, si, wj, wi
    cdef int idx = 0
    cdef double coeff

    # TODO: This value is also needed by matrix_from_mask. Loops can be
    #       reduced if needed, by sacrificing modularity and combining
    #       these functions.
    cdef int n = np.count_nonzero(mask)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] vector = np.zeros(n, dtype=np.float)

    # TODO: nogil shiz?
    for j in range(height):
        for i in range(width):

            if mask[j][i] == 0:
                continue

            # Define indices of 4-connected neighbors
            nj, ni = j - 1, i
            sj, si = j + 1, i
            ej, ei = j, i + 1
            wj, wi = j, i - 1

            # Keep a running variable that represents the
            # element of the vector. This will be assigned
            # to the array at the end.
            coeff = 0

            coeff += 4 * (source[j][i] - source[nj][ni])
            if mask[nj][ni] == 0:
                coeff += 2 * reference[nj][ni]

            coeff += 4 * (source[j][i] - source[sj][si])
            if mask[sj][si] == 0:
                coeff += 2 * reference[sj][si]

            coeff += 4 * (source[j][i] - source[ej][ei])
            if mask[ej][ei] == 0:
                coeff += 2 * reference[ej][ei]

            coeff += 4 * (source[j][i] - source[wj][wi])
            if mask[wj][wi] == 0:
                coeff += 2 * reference[wj][wi]

            # Assign the value to the output vector
            vector[idx] = coeff
            idx += 1


    return vector