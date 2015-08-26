
import numpy as np
import pyamg
from numba import jit
from scipy.sparse import csr_matrix
from geoblend.vector import create_vector



@jit()
def matrix_from_mask_numba(mask):

    height, width = mask.shape

    # Equation index and coefficient index
    eidx = 0
    cidx = 0

    # Determine the number of coefficients that will be stored
    # One approach is to convolve with a structuring element that
    # counts the number of neighbors for each pixel, but this
    # is costly.
    #
    # Another approach is to be generous when allocating the row/column/data
    # arrays by using the upper bound for the number of coefficients.
    # For N unknown pixels, there are at most 5 * N coefficients.
    n = np.count_nonzero(mask)
    n_coeff = 5 * n

    row = np.zeros(n_coeff, dtype=np.uint32)
    col = np.zeros(n_coeff, dtype=np.uint32)
    data = np.zeros(n_coeff, dtype=np.int32)

    for j in range(height):
        for i in range(width):

            if mask[j][i] == 0:
                continue

            neighbors = 0

            # Define indices of 4-connected neighbors
            nj = j - 1
            ni = i

            sj = j + 1
            si = i

            ej = j
            ei = i + 1

            wj = j
            wi = i - 1

            if nj >= 0 and nj <= height:

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

            if sj >= 0 and sj <= height:

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

            if ei >= 0 and ei <= width:

                if mask[ej][ei] == 1:

                    neighbors += 1

                    row[cidx] = eidx + 1
                    col[cidx] = eidx
                    data[cidx] = -4
                
                    cidx += 1

            if wi >= 0 and wi <= width:

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

    return csr_matrix((data[0:cidx], (row[0:cidx], col[0:cidx])), shape=(n, n))


def blend(source, reference, mask, operator, solver):
    """
    Run a Poisson blend between two arrays.
    
    :param source:
        ndarray representing the source image
    :param reference:
        ndarray representing the reference image
    :param mask:
        ndarray representing the mask
    :param operator:
        ndarray representing an operator that is applied over
        the source image. This is usually a gradient operator,
        though, others may be used.
    :param solver:
        A precomputed multilevel solver.
    """

    indices = np.nonzero(mask)

    field = (operator * source.ravel()).reshape(source.shape)
    vector = create_vector(field, reference, mask)

    x0 = source[indices].astype('float64')
    pixels = np.round(solver.solve(b=vector, x0=x0, tol=1e-05, accel='cg'))
    
    # TODO: Add dtype min/max parameter
    arr = np.zeros_like(source)
    arr[indices] = np.clip(pixels, 0, 4095)

    return arr