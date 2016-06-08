cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs


@cython.boundscheck(False)
@cython.wraparound(False)
def convolve_mask_aware(unsigned short[:, ::1] arr, char[:, ::1] mask, double threshold = 0.0):
    """
    Convolve a 2D array with a 3x3 structuring element. The convolution operates only over
    regions specified by the mask. The following kernel is used:
    
     0  -1   0
    -1   4  -1
     0  -1   0

    :param arr:
        A 2D array that will undergo convolution.
    :param mask:
        ndarray where nonzero values represent the region
        of valid pixels in an image. The mask should be
        typed to uint8.
    :param threshold:
        Difference threshold to turn off the convolution approximation.
    """

    cdef int height = mask.shape[0]
    cdef int width = mask.shape[1]

    cdef int i, j, nj, ni, sj, si, ej, ei, wj, wi, neighbors
    cdef double x, s, d
    cdef double[:, :] img = np.zeros((height, width), dtype=np.float64)

    assert arr.shape[0] == height
    assert arr.shape[1] == width

    for j in range(height):
        for i in range(width):

            if mask[j, i] == 0:
                continue

            x = 0.0
            s = <double>arr[j, i]
            neighbors = 0

            # Define indices of 4-connected neighbors
            nj = <int>(j - 1)
            ni = <int>(i)

            sj = <int>(j + 1)
            si = <int>(i)

            ej = <int>(j)
            ei = <int>(i + 1)

            wj = <int>(j)
            wi = <int>(i - 1)

            # 1. Check that the neighbor index is within image bounds
            # 2. Check that the neighbor index is within the mask

            if (nj >= 0):
                if (mask[nj, ni] != 0):
                    neighbors += 1
                    x += (-1.0 * <double>arr[nj, ni])
                else:
                    d = fabs(s - <double>arr[nj, ni])
                    if (d > threshold):
                        neighbors += 1
                        x += (-1.0 * <double>arr[nj, ni])
            
            if  (sj < height):
                if (mask[sj, si] != 0):
                    neighbors += 1
                    x += (-1.0 * <double>arr[sj, si])
                else:
                    d = fabs(s - <double>arr[sj, si])
                    if (d > threshold):
                        neighbors += 1
                        x += (-1.0 * <double>arr[sj, si])
            
            if (ei < width):
                if (mask[ej, ei] != 0):
                    neighbors += 1
                    x += (-1.0 * <double>arr[ej, ei])
                else:
                    d = fabs(s - <double>arr[ej, ei])
                    if (d > threshold):
                        neighbors += 1
                        x += (-1.0 * <double>arr[ej, ei])
            
            if (wi >= 0):
                if (mask[wj, wi] != 0):
                    neighbors += 1
                    x += (-1.0 * <double>arr[wj, wi])
                else:
                    d = fabs(s - <double>arr[wj, wi])
                    if (d > threshold):
                        neighbors += 1
                        x += (-1.0 * <double>arr[wj, wi])

            x += (<double>neighbors * <double>arr[j, i])
            img[j, i] = x

    return np.asarray(img)
