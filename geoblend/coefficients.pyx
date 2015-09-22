
import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_from_mask(char[:, ::1] mask):
    """
    Create the coefficient matrix corresponding to the
    linearized Poisson problem. This function uses a mapping
    between valid pixels specified by nonzero elements of
    a mask to construct the matrix.

    :param mask:
        ndarray (uint8) where nonzero values represent the region
        of valid pixels in an image.
    """


    cdef:
        unsigned int height = mask.shape[0]
        unsigned int width = mask.shape[1]

        # Equation index and coefficient index
        int eidx = 0
        int cidx = 0

        # Determine the number of coefficients that will be stored
        # One approach is to convolve with a structuring element that
        # counts the number of neighbors for each pixel, but this
        # is costly.
        #
        # Another approach is to be generous when allocating the row/column/data
        # arrays by using the upper bound for the number of coefficients.
        # For N unknown pixels, there are at most 5 * N coefficients.
        unsigned int n = np.count_nonzero(mask)
        unsigned int n_coeff = 5 * n

        unsigned int i, j, nj, ni, sj, si, ej, ei, wj, wi, neighbors
        unsigned int row_north, row_current, row_south
        int offset

        unsigned int[:] row = np.empty(n_coeff, dtype=np.uint32)
        unsigned int[:] col = np.empty(n_coeff, dtype=np.uint32)
        int[:] data = np.empty(n_coeff, dtype=np.int32)

        # TODO: Cast mask to boolean before this operation
        unsigned int[:] row_sum = np.sum(mask, axis=1, dtype=np.uint32)

    for j in range(1, height - 1):

        # Keep track of the nonzero counts in the north, current and south rows
        row_north = 0
        row_current = 0
        row_south = 0

        for i in range(1, width - 1):

            # Define indices of 4-connected neighbors
            nj = <unsigned int>(j - 1)
            ni = <unsigned int>(i)

            sj = <unsigned int>(j + 1)
            si = <unsigned int>(i)

            ej = <unsigned int>(j)
            ei = <unsigned int>(i + 1)

            wj = <unsigned int>(j)
            wi = <unsigned int>(i - 1)

            if mask[nj, ni] != 0:
                row_north += 1

            if mask[sj, si] != 0:
                row_south += 1

            if mask[j, i] == 0:
                continue

            neighbors = 0
            row_current += 1

            if mask[nj, ni] != 0:

                neighbors += 1

                offset = row_current - 1
                offset += row_sum[nj] - row_north + 1

                row[cidx] = eidx
                col[cidx] = eidx - offset
                data[cidx] = -4

                cidx += 1

            if mask[sj, si] != 0:

                neighbors += 1

                offset = row_south - 1
                offset += row_sum[j] - row_current + 1

                row[cidx] = eidx
                col[cidx] = eidx + offset
                data[cidx] = -4

                cidx += 1

            if mask[ej, ei] != 0:

                neighbors += 1

                row[cidx] = eidx + 1
                col[cidx] = eidx
                data[cidx] = -4

                cidx += 1

            if mask[wj, wi] != 0:

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
    return csr_matrix((data[0:cidx], (row[0:cidx], col[0:cidx])), shape=(n, n))

