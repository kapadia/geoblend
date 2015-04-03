
import rasterio as rio
from shapely.geometry.geo import box


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
    