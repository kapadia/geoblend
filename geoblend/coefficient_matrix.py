
import os
import sys
import pyamg
import numpy as np
from scipy import sparse


def save_levels(filename, levels):
    """Save levels from a PyAMG multilevel solver"""
    
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


def create_multilevel_solver(fname, matrix):
    """
    Precompute a multilevel solver for a given coefficient matrix.
    
    Each level contains a different resolution of the coefficient matrix, and
    additional matrices to assist with inversion.
    """
    shape = matrix.shape
    b = np.ones((shape[0], 1))
    ml = pyamg.smoothed_aggregation_solver(matrix, b, max_coarse=10)
    save_levels(fname, ml.levels)


def create_coefficient_matrix(shape):
    """
    Create a coefficient matrix based on the number of pixels that will be blended.
    """
    height, width = shape
    
    # Center diagonal
    c_diag = np.ones(shape)
    c_diag[1:-1, 1:-1] = 4
    c_diag = c_diag.ravel()
    
    # Upper diagonal
    u_diag = np.zeros(shape)
    u_diag[1:-1, 1:-1] = -1
    u_diag = u_diag.ravel()[:-1]
    
    # Lower diagonal
    l_diag = np.zeros(shape)
    l_diag[1:-1, 1:-1] = -1
    l_diag = l_diag.ravel()[1:]
    
    # Upper upper diagonal
    uu_diag = np.zeros(shape)
    uu_diag[1:-1, 1:-1] = -1
    uu_diag = uu_diag.ravel()[:-1*width]
    
    # Lower lower diagonal
    ll_diag = np.zeros(shape)
    ll_diag[1:-1, 1:-1] = -1
    ll_diag = ll_diag.ravel()[width:]
    
    return sparse.diags((ll_diag, l_diag, c_diag, u_diag, uu_diag), [-1 * width, -1, 0, 1, width], format='csr')

