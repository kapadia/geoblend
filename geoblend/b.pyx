
# This file is called "b" since the function below returns a column
# vector traditionally known as "b" when solving a linear system Ax=b.

import numpy as np
cimport numpy as np


DTYPE_INT8 = np.int8
DTYPE_UINT8 = np.uint8
DTYPE_UINT16 = np.uint16
ctypedef np.int8_t DTYPE_INT8_t
ctypedef np.uint8_t DTYPE_UINT8_t
ctypedef np.uint16_t DTYPE_UINT16_t


def b(mask, field, reference):
    """
    Computes the column vector b from the linearized Poisson equation.
    
    :param mask:
        ndarray where nonzero values represent the region
        of valid pixels in an image. The mask should be
        typed to uint8.
    :param field:
        The guidance field that will be preserved during blending. Usually,
        this is the gradient of the source image, though it can be any other
        vector field.
    :param reference:
        The reference image that will be used to sample for the boundary
        conditions.
    
    .. todo:: The guidance field passed has thus far been an approximation of
              the gradient. A more precise representation may be calculated here
              if the source image is passed. Plus it will save the loop required
              when convolving the source image to find the gradient field.
    """
    
    assert mask.dtype == DTYPE_UINT8
    assert field.dtype == DTYPE_INT8
    assert reference.dtype == DTYPE_UINT16
    
    # TODO: Assert shapes of mask, field, and reference are the same
    #       this ain't python. things get corrupt.
    
    cdef int height = mask.shape[0]
    cdef int width = mask.shape[1]
    
    cdef int nj, ni, ej, ei, sj, si, wj, wi, coeff
    cdef int idx = 0
    
    # TODO: This value is needed by matrix_from_mask. Loops can be
    #       reduced if needed, by sacrificing modularity by combining
    #       these functions.
    cdef int n = np.count_nonzero(mask)
    cdef np.ndarray[DTYPE_UINT16_t, ndim=1] vector = np.zeros(n, dtype=np.uint16)
    
    for j in range(height):
        for i in range(width):

            if mask[j][i] == 0:
                continue

            # Keep a running variable that represents the
            # element of the vector. This will be assigned
            # to the array at the end.
            
            # The major value is the value of the guidance field
            # at the index of the pixel. Note the negative sign.
            coeff = -field[j][i]

            nj = j - 1
            ni = i

            if mask[nj][ni] == 0:

                # Neighbor lies outside of the data region,
                # e.g. it is on the boundary. Sample the boundary
                # value from the reference image.
                coeff += reference[nj][ni]
            else:
                coeff += field[nj][ni]

            ej = j
            ei = i + 1

            if mask[ej][ei] == 0:
                coeff += reference[ej][ei]
            else:
                coeff += field[ej][ei]

            sj = j + 1
            si = i

            if mask[sj][si] == 0:
                coeff += reference[sj][si]
            else:
                coeff += field[sj][si]

            wj = j
            wi = i - 1
            if mask[wj][wi] == 0:
                coeff += reference[wj][wi]
            else:
                coeff += field[wj][wi]

            # Assign the value to the output vector
            vector[idx] = coeff
            idx += 1
    
    return vector
    
    
    
    
    
    
    
    