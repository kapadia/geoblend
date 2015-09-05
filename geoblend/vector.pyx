
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange, threadid
cimport openmp


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

    cdef int max_threads = openmp.omp_get_max_threads()
    
    cdef unsigned int[:] nj = np.empty(max_threads, np.uint32)
    cdef unsigned int[:] ni = np.empty(max_threads, np.uint32)

    cdef unsigned int[:] sj = np.empty(max_threads, np.uint32)
    cdef unsigned int[:] si = np.empty(max_threads, np.uint32)

    cdef unsigned int[:] ej = np.empty(max_threads, np.uint32)
    cdef unsigned int[:] ei = np.empty(max_threads, np.uint32)

    cdef unsigned int[:] wj = np.empty(max_threads, np.uint32)
    cdef unsigned int[:] wi = np.empty(max_threads, np.uint32)
    
    cdef double[:] coeff = np.empty(max_threads, np.float64)

    cdef int n = np.count_nonzero(mask)
    cdef double[:] vector = np.empty(n, dtype=np.float64)
    cdef unsigned long[:] row_sum = np.sum(mask, axis=1)

    assert source.shape[0] == height
    assert source.shape[1] == width
    assert reference.shape[0] == height
    assert reference.shape[1] == width

    for j in prange(height, nogil=True):
        for i in range(width):

            if mask[j, i] == 0:
                continue
            
            # Define indices of 4-connected neighbors
            nj[threadid()] = j - 1
            ni[threadid()] = i

            sj[threadid()] = j + 1
            si[threadid()] = i

            ej[threadid()] = j
            ei[threadid()] = i + 1

            wj[threadid()] = j
            wi[threadid()] = i - 1

            # Keep a running variable that represents the
            # element of the vector. This will be assigned
            # to the array at the end.
            coeff[threadid()] = 0.0

            coeff[threadid()] += 4 * (source[j, i] - source[nj[threadid()], ni[threadid()]])
            if mask[nj[threadid()], ni[threadid()]] == 0:
                coeff[threadid()] += 2 * reference[nj[threadid()], ni[threadid()]]

            coeff[threadid()] += 4 * (source[j, i] - source[sj[threadid()], si[threadid()]])
            if mask[sj[threadid()], si[threadid()]] == 0:
                coeff[threadid()] += 2 * reference[sj[threadid()], si[threadid()]]

            coeff[threadid()] += 4 * (source[j, i] - source[ej[threadid()], ei[threadid()]])
            if mask[ej[threadid()], ei[threadid()]] == 0:
                coeff[threadid()] += 2 * reference[ej[threadid()], ei[threadid()]]

            coeff[threadid()] += 4 * (source[j, i] - source[wj[threadid()], wi[threadid()]])
            if mask[wj[threadid()], wi[threadid()]] == 0:
                coeff[threadid()] += 2 * reference[wj[threadid()], wi[threadid()]]

            # Assign the value to the output vector
            vector[row_sum[j] + i] = coeff[threadid()]

    return np.asarray(vector)
