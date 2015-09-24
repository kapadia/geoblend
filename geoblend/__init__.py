
import numpy as np
from geoblend.vector import create_vector


def blend(source, reference, mask, solver, gradient_multiplier):
    """
    Run a Poisson blend between two arrays.
    
    :param source:
        ndarray representing the source image
    :param reference:
        ndarray representing the reference image
    :param mask:
        ndarray representing the mask
    :param solver:
        A precomputed multilevel solver.
    """

    indices = np.nonzero(mask)

    vector = create_vector(source.astype(np.float64), reference.astype(np.float64), mask, multiplier=gradient_multiplier)

    x0 = source[indices].astype('float64')
    pixels = np.round(solver.solve(b=vector, x0=x0, tol=1e-03, accel='cg'))
    
    # TODO: Add dtype min/max parameter
    arr = np.zeros_like(source)
    arr[indices] = np.clip(pixels, 0, 4095)

    return arr