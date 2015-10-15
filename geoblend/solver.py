
import numpy as np
from scipy import sparse

import pyamg
from pyamg.relaxation.smoothing import change_smoothers


def load(file):
    """
    Deserialize a file containing various numpy matrices to
    a PyAMG multigrid solver.

    :param file:
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

    data = np.load(file)
    levels = map(load_level, data["levels"])

    ml = pyamg.multilevel.multilevel_solver(levels, coarse_solver='pinv2')
    change_smoothers(ml, 'gauss_seidel', 'gauss_seidel')

    return ml


def save(file, solver):
    """
    Serialize a PyAMG multigrid solver to disk using numpy's
    binary format.
    
    :param file:
    :param solver:
        Multi-level PyAMG solver.
    """

    def export_level(level):

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

    output = map(export_level, solver.levels)
    np.savez(file, levels=output)
