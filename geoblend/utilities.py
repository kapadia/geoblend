
import numpy as np
import rasterio as rio

from skimage.morphology import disk, binary_erosion


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
                mask = src.read(4).astype(np.bool)
                mask = binary_erosion(mask, disk(3)).astype(np.uint8)

                # height, width = src.shape
                # h2, w2 = height / 2, width / 2
                # mask = np.zeros(src.shape, dtype=np.uint8)
                # r = 600
                # mask[h2 - r: h2 + r + 1, w2 - r: w2 + r + 1] = disk(r)

    return mask