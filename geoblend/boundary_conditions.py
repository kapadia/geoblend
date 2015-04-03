
import pyamg
from utilities import *


def dirichlet(func):
    """
    Decorator used by the Poisson blend function to set boundary conditions.
    
    This sets Dirichlet (i.e. constant) boundary conditions extracted from
    the target image.
    
    :param func:
        geoblend.poisson_blend
    
    """
    
    def inner(srcpath, tarpath, bidx, solver=None, operator=None, boundary_condition=None):
        
        source_bounds = get_bounds(srcpath)
        target_bounds = get_bounds(tarpath)
        intersect_bounds = get_intersection(source_bounds, target_bounds)
        
        source_window = get_window(srcpath, intersect_bounds)
        target_window = get_window(tarpath, intersect_bounds)
        
        boundary_condition = {}
        for side in ['top', 'right', 'bottom', 'left']:
            window = get_boundary_window(target_window, side=side)
            boundary_condition[side] = get_band(tarpath, bidx, window).ravel()
        
        return func(srcpath, tarpath, bidx, solver=solver, operator=operator, boundary_condition=boundary_condition)
        
    return inner