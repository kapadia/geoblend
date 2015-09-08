
import numpy as np
import rasterio as rio
from skimage.morphology import binary_erosion, disk


def get_mask(srcpath):
    """
    Read the mask for a given filepath. It is assumed
    that the 4th band corresponds to a mask. If a 4th
    band does not exist, a square mask is generated with
    border pixels marked out.

    :param srcpath:
    """

    with rio.drivers():
        with rio.open(srcpath) as src:
            count = src.count

            if count < 4:
                mask = np.ones(src.shape, dtype=np.uint8)
                mask[0, :] = 0
                mask[-1, :] = 0
                mask[:, 0] = 0
                mask[:, -1] = 0
            else:
                mask = src.read(4)
                mask[np.nonzero(mask)] = 1
                print "10"
                mask = binary_erosion(mask, disk(10)).astype(np.uint8)
                print "EROSION DONE"

    return mask