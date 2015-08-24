
import numpy as np
import pyamg
from geoblend.vector import create_vector


def blend(source, reference, mask, operator, solver):
    """
    Run a Poisson blend between two arrays. A solver should be passed
    to this function.
    
    :param source:
        ndarray representing the source image
    :param reference:
        ndarray representing the reference image
    :param mask:
        ndarray representing the mask
    :param operator:
        ndarray representing an operator that is applied over
        the source image. This is usually a gradient operator,
        though, others may be used.
    :param solver:
        A precomputed multilevel solver.
    """

    indices = np.nonzero(mask)

    field = (operator * source.ravel()).reshape(source.shape)
    vector = create_vector(field, reference, mask)

    x0 = source[indices].astype('float64')
    pixels = np.round(solver.solve(b=vector, x0=x0, tol=1e-16))
    
    # TODO: Add dtype min/max parameter
    arr = np.zeros_like(source)
    arr[indices] = np.clip(pixels, 0, 4095)

    return arr


class ProjectionMismatchError(Exception):
    pass

class IntersectionError(Exception):
    pass

class BandCountMismatchError(Exception):
    pass

class DataTypeMismatchError(Exception):
    pass

class SolverRequiredError(Exception):
    pass

class OperatorRequiredError(Exception):
    pass
    
class BoundaryConditionError(Exception):
    pass
