
import pyamg
import numpy as np
from scipy import sparse


def save_multilevel_solver(filename, levels):
    """
    Save levels from a PyAMG multilevel solver

    :param filename:
    :param levels:
        PyAMG multilevel solver
    """

    def restructure_level_for_export(level):

        matrix_keys = filter(lambda k: k in ['A', 'B', 'P', 'R'], level.__dict__.keys())
        obj = {}

        for key in matrix_keys:
            array = getattr(level, key)

            if type(array) == np.ndarray:
                obj[key] = array
            else:
                obj[key] = {
                    "data": array.data,
                    "indices": array.indices,
                    "indptr": array.indptr,
                    "shape": array.shape,
                    "format": array.format
                }

        # TODO: Save the configuration of the pre and postsmoother functions
        # if 'presmoother' in level.__dict__.keys():
        #     obj['presmoother'] = {}

        return obj

    output = map(restructure_level_for_export, levels)
    np.savez(filename, levels=output)


def load_multilevel_solver(filename):
    """
    Load a multilevel solver from disk.

    :param filename:
    """

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

    data = np.load(filename)
    return map(load_level, data["levels"])


def create_multilevel_solver(filename, matrix):
    """
    Precompute a multilevel solver for a given coefficient matrix.
    
    Each level contains a different resolution of the coefficient matrix, and
    additional matrices to assist with inversion.
    
    :param filename:
    :param matrix:
        CSR sparse matrix
    """

    shape = matrix.shape
    b = np.ones((shape[0], 1))
    ml = pyamg.smoothed_aggregation_solver(matrix, b, max_coarse=10)
    
    save_multilevel_solver(filename, ml.levels)
