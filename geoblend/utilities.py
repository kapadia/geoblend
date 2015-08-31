
import numpy as np
import rasterio as rio
from skimage.morphology import binary_erosion, square, convex_hull_image, disk
from skimage.filters import gaussian_filter
# from scipy.ndimage.filters import gaussian_filter


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
                mask = np.ones(src.shape)
                mask[0, :] = 0
                mask[-1, :] = 0
                mask[:, 0] = 0
                mask[:, -1] = 0
            else:
                mask = src.read(4).astype(np.uint8)
                mask[np.nonzero(mask)] = 1
                mask = binary_erosion(convex_hull_image(mask), disk(3)).astype(np.uint8)

    return mask