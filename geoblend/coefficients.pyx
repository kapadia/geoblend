
import numpy as np
cimport numpy as np


DTYPE_INT8 = np.int8
DTYPE_UINT8 = np.uint8
DTYPE_INT32 = np.int32
DTYPE_UINT32 = np.uint32

ctypedef np.int8_t DTYPE_INT8_t
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

    .. todo:: Support masks that represent the rectangular image case.
    .. todo:: Support valid pixels on the edge of the mask. Currently,
              padding is required between valid data and the mask edge.
    """

    assert mask.dtype == DTYPE_UINT8

    cdef int height = mask.shape[0]
    cdef int width = mask.shape[1]

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
    cdef int n = np.count_nonzero(mask)
    cdef int n_coeff = 5 * n

    cdef np.ndarray[DTYPE_UINT32_t, ndim=1] row = np.zeros(n_coeff, dtype=np.uint32)
    cdef np.ndarray[DTYPE_UINT32_t, ndim=1] col = np.zeros(n_coeff, dtype=np.uint32)
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] data = np.zeros(n_coeff, dtype=np.int32)
    
    cdef int i, j, ii, neighbors, nj, ni, ej, ei, sj, si, wj, wi, offset

    for j in range(height):
        for i in range(width):

            if mask[j][i] == 0:
                continue

            neighbors = 0

            # Define indices of 4-connected neighbors
            nj, ni = j - 1, i
            sj, si = j + 1, i
            ej, ei = j, i + 1
            wj, wi = j, i - 1

            if nj >= 0 and nj <= height:

                neighbors += 1

                if mask[nj][ni] == 1:

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

            if sj >= 0 and sj <= height:

                neighbors += 1

                if mask[sj][si] == 1:

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

            if ei >= 0 and ei <= width:

                neighbors += 1

                if mask[ej][ei] == 1:
                
                    row[cidx] = eidx + 1
                    col[cidx] = eidx
                    data[cidx] = -4
                
                    cidx += 1
            
            if wi >= 0 and wi <= width:

                neighbors += 1

                if mask[wj][wi] == 1:
                
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

