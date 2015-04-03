
import os
import logging

import numpy as np
from scipy import sparse
from scipy.misc import imresize

import pyamg
from pyamg.relaxation.smoothing import change_smoothers

import rasterio as rio

from boundary_conditions import dirichlet
from utilities import *
from geoblend import *
from coefficient_matrix import create_coefficient_matrix, create_multilevel_solver

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def load_levels(f):
    
    class level:
    
        def __init__(self):
            pass
    
    
    def load_level(item):
    
        l = level()
        for key, value in item.iteritems():
        
            if key == 'presmoother':
                continue
            
            if type(value) is np.ndarray:
                arr = value
            else:
                matrix_func_name = "%s_matrix" % value["format"]
                matrix_func = getattr(sparse, matrix_func_name)
                arr = matrix_func((
                    value["data"],
                    value["indices"],
                    value["indptr"]),
                    shape=value["shape"]
                )
        
            setattr(l, key, arr)
    
        return l
    
    data = np.load(f)
    return map(load_level, data["levels"])


@dirichlet
def poisson_blend(srcpath, tarpath, bidx, solver=None, operator=None, boundary_condition=None):
    
    if solver is None:
        raise SolverRequiredError
    if operator is None:
        raise OperatorRequiredError
    if boundary_condition is None:
        raise BoundaryConditionError
    
    source_bounds = get_bounds(srcpath)
    target_bounds = get_bounds(tarpath)
    intersect_bounds = get_intersection(source_bounds, target_bounds)
    
    # Get the corresponding windows w.r.t to the source bounds
    source_window = get_window(srcpath, intersect_bounds)
    target_window = get_window(tarpath, intersect_bounds)
    
    # Get the shape from the target window
    shape = tuple(map(lambda p: p[1] - p[0], target_window))
    height, width = shape
    
    # Get source pixels inside of the window
    source = get_band(srcpath, bidx, source_window)
    dtype_info = np.iinfo(source.dtype)
    
    # Often there are subtle resolutions differences between
    # source and target using the same bounds due to projection
    # rounding errors. Ensure the source image is the same shape
    # as the target image.
    
    if source.shape != shape:
        # TODO: Use zoom instead.
        source = imresize(source, shape, interp='nearest')
    
    # Get the gradient of the source image using the operator
    guidance_field = operator * source.ravel()
    guidance_field = guidance_field.reshape(source.shape)
    
    guidance_field[0, :] = boundary_condition["top"]
    guidance_field[:, -1] = boundary_condition["right"]
    guidance_field[-1, :] = boundary_condition["bottom"]
    guidance_field[:, 0] = boundary_condition["left"]
    
    guidance_field = guidance_field.ravel()
    
    x0 = np.ravel(source).astype('float64')
    x = solver.solve(b=guidance_field, x0=x0, tol=1e-05)
    
    return np.clip(x, dtype_info.min, dtype_info.max).reshape(shape)


def blend(srcpath, tarpath, dstpath):
    """
    Geo-aware Poisson blend to smoothly apply the source image
    over the target image.
    
    :param srcpath:
        Source image path.
        
    :param tarpath:
        Target image path.
        
    :param dstpath:
        Output image path.
    """
    
    source_metadata = get_metadata(srcpath)
    target_metadata = get_metadata(tarpath)
    
    source_bounds = get_bounds(srcpath)
    target_bounds = get_bounds(tarpath)
    intersect_bounds = get_intersection(source_bounds, target_bounds)
    
    # Get the corresponding window in the target image
    target_window = get_window(tarpath, intersect_bounds)
    source_window = get_window(srcpath, intersect_bounds)
    
    if source_metadata["crs"] != target_metadata["crs"]:
        raise ProjectionMismatchError("Projections of %s and %s do not match." % (srcpath, tarpath))
    
    if not intersect_bounds:
        raise IntersectionError("No intersecting pixels in %s and %s to blend." % (srcpath, tarpath))
    
    if source_metadata["count"] != target_metadata["count"]:
        raise BandCountMismatchError("Band count for %s and %s must be the same." % (srcpath, tarpath))
    
    if source_metadata["dtype"] != target_metadata["dtype"]:
        raise DataTypeMismatchError("Data type for %s and %s must be the same." % (srcpath, tarpath))
    
    shape = tuple(map(lambda p: p[1] - p[0], target_window))
    
    # Set up a few paths
    module_path = os.path.join(os.path.dirname(__file__), '..')
    matrices_path = os.path.join(module_path, "data", "matrices")
    matrix_path = os.path.join(matrices_path, "%dx%d.npz" % shape)
    
    if not os.path.exists(matrix_path):
        
        if not os.path.exists(matrices_path):
            os.makedirs(matrices_path)
        
        logging.info("Constructing coefficient matrix. This will be slow for large images.")
        matrix = create_coefficient_matrix(shape)
        create_multilevel_solver(matrix_path, matrix)
    
    levels = load_levels(matrix_path)
    
    logging.info("Constructing a multi-level solver")
    ml = pyamg.multilevel.multilevel_solver(levels, coarse_solver='pinv2')
    
    change_smoothers(ml, 'gauss_seidel', 'gauss_seidel')
    
    # The guidance field will be the gradient of the source image. This will
    # preserve the gradient in the model image. A poisson kernal is used to
    # compute the field. Other guidance fields could be used in place of the
    # gradient.
    P = pyamg.gallery.poisson(shape)
    
    count = min(source_metadata["count"], target_metadata["count"])
    with rio.drivers():
        with rio.open(dstpath, 'w', **target_metadata) as dst:
            
            for bidx in range(1, count + 1):
                logging.info("Blending channel %d" % bidx)
                
                patch = poisson_blend(srcpath, tarpath, bidx, solver=ml, operator=P)
                band = get_band(tarpath, bidx, None)
                
                ((y0, y1), (x0, x1)) = target_window
                band[y0:y1, x0:x1] = patch
                
                dst.write_band(bidx, band)

