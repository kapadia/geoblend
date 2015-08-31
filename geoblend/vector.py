

import numpy as np
import pyamg
from scipy.ndimage import convolve


def create_vector(field, reference, mask):
    """
    Create the RHS of the linearized Poisson problem based on
    the gradient of the source image and the boundaries of the
    reference image.

    .. todo:: Current implementation results in something wacky.
    """

    indices = np.nonzero(mask)

    # TODO: There might be a better structuring element for this case.
    #
    #       e.g. [0, 1, 0]
    #            [1, 0, 1]
    #            [0, 1, 0]
    #
    #       This saves once selection, over the 2 below.
    selem = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    # TODO: Establish some conventions so that casting the mask
    #       to float isn't needed.
    m = convolve(mask.astype(np.float), selem, mode='constant', cval=0)
    m[m < 0] = 0
    m[m > 0] = 1
    bindices = np.nonzero(m)

    field[bindices] = reference[bindices]

    return field[indices].astype('float64')
