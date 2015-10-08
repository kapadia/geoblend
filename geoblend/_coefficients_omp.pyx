
import numpy as np
from scipy.sparse import csr_matrix

cimport numpy as np
cimport cython
from cython.parallel cimport prange, threadid
cimport openmp


cdef inline unsigned int[:] count_nonzero_along_axis(char[:, ::1] arr):
    
    cdef:
        unsigned int i, j, count
        unsigned int height = arr.shape[0]
        unsigned int width = arr.shape[1]
        unsigned int[:] row_sum = np.empty(height, dtype=np.uint32)
    
    for j in range(height):
        count = 0
        for i in range(width):
            if arr[j, i] != 0:
                count += 1
            if i == width - 1:
                row_sum[j] = count

    return np.asarray(row_sum)


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

        int max_threads = openmp.omp_get_max_threads()

        # Equation index and coefficient index
        unsigned int[:] eidx = np.zeros(max_threads, np.uint32)
        unsigned int[:] cidx = np.zeros(max_threads, np.uint32)

        # Determine the number of coefficients that will be stored
        # One approach is to convolve with a structuring element that
        # counts the number of neighbors for each pixel, but this
        # is costly.
        #
        # Another approach is to be generous when allocating the row/column/data
        # arrays by using the upper bound for the number of coefficients.
        # For N unknown pixels, there are at most 5 * N coefficients.
        unsigned int[:] row_sum = count_nonzero_along_axis(mask)
        unsigned int n = np.sum(row_sum)
        unsigned int n_coeff = 5 * n

        unsigned int i, j

        unsigned int[:] nj = np.zeros(max_threads, np.uint32)
        unsigned int[:] ni = np.zeros(max_threads, np.uint32)
        unsigned int[:] sj = np.zeros(max_threads, np.uint32)
        unsigned int[:] si = np.zeros(max_threads, np.uint32)
        unsigned int[:] ej = np.zeros(max_threads, np.uint32)
        unsigned int[:] ei = np.zeros(max_threads, np.uint32)
        unsigned int[:] wj = np.zeros(max_threads, np.uint32)
        unsigned int[:] wi = np.zeros(max_threads, np.uint32)

        unsigned int[:] neighbors = np.zeros(max_threads, np.uint32)

        unsigned int[:] row_north = np.zeros(max_threads, np.uint32)
        unsigned int[:] row_current = np.zeros(max_threads, np.uint32)
        unsigned int[:] row_south = np.zeros(max_threads, np.uint32)

        unsigned int i, j, nj, ni, sj, si, ej, ei, wj, wi, neighbors
        unsigned int row_north, row_current, row_south

        int[:] offset = np.zeros(max_threads, np.int32)

        unsigned int[:] row = np.empty(n_coeff, dtype=np.uint32)
        unsigned int[:] col = np.empty(n_coeff, dtype=np.uint32)
        int[:] data = np.empty(n_coeff, dtype=np.int32)

    for j in prange(1, height - 1, nogil=True):

        # Keep track of the nonzero counts in the north, current and south rows
        row_north[threadid()] = 0
        row_current[threadid()] = 0
        row_south[threadid()] = 0

        for i in range(1, width - 1):

            # Define indices of 4-connected neighbors
            nj[threadid()] = <unsigned int>(j - 1)
            ni[threadid()] = <unsigned int>(i)

            sj[threadid()] = <unsigned int>(j + 1)
            si[threadid()] = <unsigned int>(i)

            ej[threadid()] = <unsigned int>(j)
            ei[threadid()] = <unsigned int>(i + 1)

            wj[threadid()] = <unsigned int>(j)
            wi[threadid()] = <unsigned int>(i - 1)

            if mask[nj, ni] != 0:
                row_north[threadid()] += 1

            if mask[sj, si] != 0:
                row_south[threadid()] += 1

            if mask[j, i] == 0:
                continue

            neighbors[threadid()] = 0
            row_current[threadid()] += 1

            if mask[nj[threadid()], ni[threadid()]] != 0:

                neighbors[threadid()] += 1

                offset[threadid()] = row_current[threadid()] - 1
                offset[threadid()] += row_sum[nj[threadid()]] - row_north[threadid()] + 1

                row[cidx] = eidx
                col[cidx] = eidx - offset[threadid()]
                data[cidx] = -4

                cidx += 1

            if mask[sj[threadid()], si[threadid()]] != 0:

                neighbors[threadid()] += 1

                offset[threadid()] = row_south[threadid()] - 1
                offset[threadid()] += row_sum[j] - row_current[threadid()] + 1

                row[cidx] = eidx
                col[cidx] = eidx + offset[threadid()]
                data[cidx] = -4

                cidx += 1

            if mask[ej[threadid()], ei[threadid()]] != 0:

                neighbors[threadid()] += 1

                row[cidx] = eidx + 1
                col[cidx] = eidx
                data[cidx] = -4

                cidx += 1

            if mask[wj[threadid()], wi[threadid()]] != 0:

                neighbors[threadid()] += 1

                row[cidx] = eidx - 1
                col[cidx] = eidx
                data[cidx] = -4

                cidx += 1

            row[cidx] = eidx
            col[cidx] = eidx
            data[cidx] = 2 * neighbors[threadid()] + 8

            # Increment the equation index
            cidx += 1
            eidx += 1

    indices = np.nonzero(data)
    return csr_matrix((data[indices], (row[indices], col[indices])), shape=(n, n))

