cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs, modf


@cython.boundscheck(False)
@cython.wraparound(False)
def convolve_along_boundaries(float[:, ::1] arr, char[:, ::1] st):
    """
    Apply a convolution using a median filter only along mask boundaries.

    :param arr:
        A 2D array that will undergo partial convolution.
    :param st:
        Source trace file.
    """

    cdef int height = st.shape[0]
    cdef int width = st.shape[1]

    assert arr.shape[0] == height
    assert arr.shape[1] == width

    cdef float neighbors
    cdef float valid_neighbors
    cdef float x

    cdef int i, j, ni, nj, ei, ej, si, sj, wi, wj
    cdef int dx, dy, ii, jj
    cdef int stidx

    cdef float[:, :] out = np.zeros((height, width), dtype=np.float32)

    for j in range(height):
        for i in range(width):
        
            # Get the source trace index for the current pixel
            stidx = st[j, i]

            neighbors = 0.0
            valid_neighbors = 0.0
            x = 0.0

            for dy in range(-1, 2):
                for dx in range(-1, 2):

                    ii = <int>(i + dx)
                    jj = <int>(j + dy)

                    if (ii == 0) and (jj == 0):
                        continue

                    if (jj >= 0) and (jj < height):
                        if (ii >= 0) and (ii < width):

                            neighbors += 1
                            x += arr[jj, ii]

                            if (st[jj, ii] == stidx):
                                valid_neighbors += 1

            if (neighbors == valid_neighbors):
                out[j, i] = arr[j, i]
            else:
                out[j, i] = (x / neighbors)

    return np.asarray(out)



@cython.boundscheck(False)
@cython.wraparound(False)
def nearest_mask_aware(double[:, ::1] arr, char[:, ::1] mask):
    """
    Upsample a 2D array by a factor of two using nearest neighbor
    and respecting the bounds of a mask.

    :param arr:
        A 2D array that will be upsampled by a factor of 2.
    :param mask:
        ndarray where nonzero values represent the region
        of valid pixels in an image. The mask should be
        typed to uint8.
    """

    cdef int height = mask.shape[0]
    cdef int width = mask.shape[1]

    assert arr.shape[0] == height
    assert arr.shape[1] == width

    cdef int new_height = 2 * height
    cdef int new_width = 2 * width

    cdef int i, j, s0, s1, t0, t1
    cdef double p00, p10, p01, p11
    cdef double a00, a10, a01, a11
    cdef double xi, xf, yi, yf
    cdef double n, val

    cdef double[:, :] img = np.zeros((new_height, new_width), dtype=np.float64)

    for j in range(new_height):
        for i in range(new_width):

            # Define indices for pixels in the nearest 2x2 neighborhood
            s0 = <int>(<float>i / 2.0)
            t0 = <int>(<float>j / 2.0)

            if mask[t0, s0] == 0:
                continue

            s1 = <int>(<float>i / 2.0 + 1.0)
            t1 = <int>(<float>j / 2.0 + 1.0)
            n = 1.0

            val = arr[t0, s0]
            if (s1 <= width):
                if mask[t0, s1] != 0:
                    n += 1
                    val += arr[t0, s1]

            if (t1 <= height):
                if mask[t1, s0] != 0:
                    n += 1
                    val += arr[t1, s0]

            if (s1 <= width) and (t1 <= height):
                if mask[t1, s1] != 0:
                    n += 1
                    val += arr[t1, s1]

            img[j, i] = (val / n)

    return np.asarray(img)


@cython.boundscheck(False)
@cython.wraparound(False)
def resample_mask_aware(double[:, ::1] arr, char[:, ::1] mask, char[:, ::1] mask_decimated):
    """
    Upsample a 2D array by a factor of two using bilinear interpolation
    and respecting the bounds of a mask.

    :param arr:
        A 2D array that will be upsampled by a factor of 2.
    :param mask:
        ndarray where nonzero values represent the region
        of valid pixels in an image. The mask should be
        typed to uint8.
    :param mask_decimated:
    """

    cdef int height = mask.shape[0]
    cdef int width = mask.shape[1]

    cdef int decimated_height = mask_decimated.shape[0]
    cdef int decimated_width = mask_decimated.shape[1]

    assert arr.shape[0] == decimated_height
    assert arr.shape[1] == decimated_width

    cdef int x0, x1, y0, y1
    cdef int s0, s1, t0, t1
    cdef double p00, p10, p01, p11
    cdef double a00, a10, a01, a11
    cdef double xi, xf, yi, yf
    cdef double neighbors
    cdef double x

    cdef int s, t, ns, nt, es, et, ss, st, ws, wt

    cdef double[:, :] img = np.zeros((height, width), dtype=np.float64)

    for y0 in range(height):
        for x0 in range(width):

            if mask[y0, x0] == 0:
                continue

            # Define indices for pixels in the decimated reference frame
            s = <int>(<float>x0 / 2.0)
            t = <int>(<float>y0 / 2.0)

            if mask_decimated[t, s] == 0:
                continue

            ns = <int>(s)
            nt = <int>(t - 1)

            es = <int>(s + 1)
            et = <int>(t)

            ss = <int>(s)
            st = <int>(t + 1)

            ws = <int>(s - 1)
            wt = <int>(t)

            neighbors = 0.0
            x = arr[t, s]

            if (nt >= 0):
                if (mask_decimated[nt, ns] != 0):
                    neighbors += 1
                    x += arr[nt, ns]

            if (es < decimated_width):
                if (mask_decimated[et, es] != 0):
                    neighbors += 1
                    x += arr[et, es]

            if (st < decimated_height):
                if (mask_decimated[st, ss] != 0):
                    neighbors += 1
                    x += arr[st, ss]

            if (ws >= 0):
                if (mask_decimated[wt, ws] != 0):
                    neighbors += 1
                    x += arr[wt, ws]

            img[y0, x0] = (x / (neighbors + 1.0))

    return np.asarray(img)


@cython.boundscheck(False)
@cython.wraparound(False)
def bilinear_mask_aware(double[:, ::1] arr, char[:, ::1] mask):
    """
    Upsample a 2D array by a factor of two using bilinear interpolation
    and respecting the bounds of a mask.

    :param arr:
        A 2D array that will be upsampled by a factor of 2.
    :param mask:
        ndarray where nonzero values represent the region
        of valid pixels in an image. The mask should be
        typed to uint8.
    """

    cdef int height = mask.shape[0]
    cdef int width = mask.shape[1]

    cdef int decimated_height = <int>(0.5 * height)
    cdef int decimated_width = <int>(0.5 * width)

    assert arr.shape[0] == decimated_height
    assert arr.shape[1] == decimated_width

    cdef int x0, x1, y0, y1
    cdef int s0, s1, t0, t1
    cdef double p00, p10, p01, p11
    cdef double a00, a10, a01, a11
    cdef double xi, xf, yi, yf

    cdef double[:, :] img = np.zeros((height, width), dtype=np.float64)

    for y0 in range(height):
        for x0 in range(width):

            if mask[y0, x0] == 0:
                continue

            # Define indices for pixels in the nearest 2x2 neighborhood
            x1 = <int>(x0 + 1)
            y1 = <int>(y0 + 1)

            s0 = <int>(<float>x0 / 2.0)
            t0 = <int>(<float>y0 / 2.0)

            if (s0 >= decimated_width):
                continue
            if (t0 >= decimated_height):
                continue

            s1 = <int>(<float>x0 / 2.0 + 1.0)
            t1 = <int>(<float>y0 / 2.0 + 1.0)

            p00 = arr[t0, s0]

            if (x1 == width):
                p10 = arr[t0, s0]
            elif (mask[y0, x1] == 0):
                p10 = arr[t0, s0]
            else:
                # Is there any way for s1 to exceed the decimated width?
                p10 = arr[t0, s1]
            
            if (y1 == height):
                p01 = arr[t0, s0]
            elif (mask[y1, x0] == 0):
                p01 = arr[t0, s0]
            else:
                # Is there any way for t1 to exceed the decimated height?
                p01 = arr[t1, s0]

            p11 = arr[t0, s0]
            if (x1 == width) and (y1 == height):
                p11 = arr[t0, s0]
            elif (mask[y1, x1] == 0):
                p11 = arr[t0, s0]
            else:
                if (s1 == decimated_width):
                    s1 = s0
                if (t1 == decimated_height):
                    t1 = t0
                p11 = arr[t0, s0]

            a00 = p00
            a10 = p10 - p00
            a01 = p01 - p00
            a11 = p11 + p00 - p10 - p01

            xf = modf(<float>x0 / 2.0, &xi)
            yf = modf(<float>y0 / 2.0, &yi)

            img[y0, x0] = a00 + a10 * xf + a01 * yf + a11 * xf * yf

    return np.asarray(img)
