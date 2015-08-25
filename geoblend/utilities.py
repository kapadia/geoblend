
import numpy as np
import rasterio as rio
from shapely.geometry.geo import box
from skimage.morphology import binary_erosion, square


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
                mask = np.ones_like(src.shape)
                mask[0, :] = 0
                mask[-1, :] = 0
                mask[:, 0] = 0
                mask[:, -1] = 0
            else:
                mask = src.read(4)
                return binary_erosion(mask, square(4)).astype(np.uint8)

    return mask


def get_metadata(fpath):
    """
    Read metadata from an image.
    
    :param fpath:
        Filepath to image.
    """
    
    with rio.drivers():
        with rio.open(fpath) as src:
            metadata = src.meta
    
    return metadata


def get_bounds(fpath):
    """
    Get the geospatial bounds of an image.
    
    :param fpath:
        Filepath to image.
    """
    
    with rio.drivers():
        with rio.open(fpath, 'r') as src:
            bounds = src.bounds
    
    return bounds


def get_window(fpath, bounds):
    """
    Get the window in pixel units for a given geospatial boundary.
    
    :param fpath:
        Filepath to image.
    :param bounds:
        Tuple representing geospatial bounds as (w, s, e ,n)
    """
    
    with rio.drivers():
        with rio.open(fpath, 'r') as src:
            window = src.window(*bounds)
    
    return window


def get_intersection(bounds1, bounds2):
    """
    Get the intersection of two bounding boxes.
    
    :param bounds1:
        Bounding box in any units.
    :param bounds2:
        Bounding box in any units.
    """
    bbox1 = box(*bounds1)
    bbox2 = box(*bounds2)
    
    return bbox1.intersection(bbox2).bounds


def get_band(fpath, bidx, window):
    """
    Read a raster band from an image path.
    
    :param fpath:
        Filepath to image.
    :param bidx:
        Band index.
    :param window:
        Tuple representing a window in units of pixels ((row_start, row_stop), (col_start, col_stop))
    """
    
    with rio.drivers():
        with rio.open(fpath, 'r') as src:
            band = src.read(bidx, window=window, masked=False)
    
    return band


def get_boundary_window(window, side=None, extrude=1):
    """
    Determine the boundary indicies for a given window.
    
    :param window:
        Tuple representing indices of a 2d array.
    :param side:
        The side of a box. Options are top, bottom, left, right.
    :param extrude:
        The number of pixels to extrude.
    """
    ((y0, y1), (x0, x1)) = window
    
    return {
        'top': ((y0, y0 + extrude), (x0, x1)),
        'bottom': ((y1 - extrude, y1), (x0, x1)),
        'left': ((y0, y1), (x0, x0 + extrude)),
        'right': ((y0, y1), (x1 - extrude, x1)),
        None: window
    }[side]
    