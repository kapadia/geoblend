
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython


DTYPE_UINT8 = np.uint8
DTYPE_INT32 = np.int32
DTYPE_UINT32 = np.uint32

ctypedef np.uint8_t DTYPE_UINT8_t
ctypedef np.int32_t DTYPE_INT32_t
ctypedef np.uint32_t DTYPE_UINT32_t


def matrix_from_mask(np.ndarray[DTYPE_UINT8_t, ndim=2] mask):
    """
    Create the coefficient matrix corresponding to the
    linearized Poisson problem. This function uses a mapping
    between valid pixels specified by nonzero elements of
    a mask to construct the matrix.

    :param mask:
        ndarray where nonzero values represent the region
        of valid pixels in an image. The mask should be
        typed to uint8.

    .. todo:: Support valid pixels on the edge of the mask. Currently,
              padding is required between valid data and the mask edge.
    """

    assert mask.dtype == DTYPE_UINT8

    cdef unsigned int height = mask.shape[0]
    cdef unsigned int width = mask.shape[1]

    # Equation index and coefficient index
    cdef int eidx = 0
    cdef int cidx = 0

    # Determine the number of coefficients that will be stored
    # One approach is to convolve with a structuring element that
    # counts the number of neighbors for each pixel, but this
    # is costly.
    #
    # Another approach is to be generous when allocating the row/column/data
    # arrays by using the upper bound for the number of coefficients.
    # For N unknown pixels, there are at most 5 * N coefficients.
    cdef unsigned int n = np.count_nonzero(mask)
    cdef unsigned int n_coeff = 5 * n

    cdef np.ndarray[DTYPE_UINT32_t, ndim=1] row = np.zeros(n_coeff, dtype=np.uint32)
    cdef np.ndarray[DTYPE_UINT32_t, ndim=1] col = np.zeros(n_coeff, dtype=np.uint32)
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] data = np.zeros(n_coeff, dtype=np.int32)

    cdef unsigned int i, j, ii, nj, ni, sj, si, ej, ei, wj, wi, neighbors
    cdef int offset

    for j in range(height):
        for i in range(width):

            if mask[j][i] == 0:
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

            if nj <= height:

                if mask[nj][ni] == 1:

                    neighbors += 1

                    # Count the number of valued pixels in the previous 
                    # row, and current row.
                    # BT-dubs - this is less efficient than I'd prefer.
                    offset = 0
                    for ii in range(ni, width):
                        if mask[nj][ii] == 1:
                            offset += 1
                    for ii in range(0, i):
                        if mask[j][ii] == 1:
                            offset += 1
                
                    row[cidx] = eidx
                    col[cidx] = eidx - offset
                    data[cidx] = -4
                
                    cidx += 1

            if sj <= height:

                if mask[sj][si] == 1:

                    neighbors += 1

                    offset = 0
                    for ii in range(i, width):
                        if mask[j][ii] == 1:
                            offset += 1
                    for ii in range(0, si):
                        if mask[sj][ii] == 1:
                            offset += 1

                    row[cidx] = eidx
                    col[cidx] = eidx + offset
                    data[cidx] = -4

                    cidx += 1

            if ei <= width:

                if mask[ej][ei] == 1:

                    neighbors += 1

                    row[cidx] = eidx + 1
                    col[cidx] = eidx
                    data[cidx] = -4
                
                    cidx += 1
            
            if wi <= width:

                if mask[wj][wi] == 1:

                    neighbors += 1

                    row[cidx] = eidx - 1
                    col[cidx] = eidx
                    data[cidx] = -4
                
                    cidx += 1

            row[cidx] = eidx
            col[cidx] = eidx
            data[cidx] = 2 * neighbors + 8
            
            # Increment the equation index
            cidx += 1
            eidx += 1

    # Return a slice since the allocation was an approximation
    return data[0:cidx], row[0:cidx], col[0:cidx], n, n

