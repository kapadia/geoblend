
import numpy as np
from scipy import sparse
from geoblend.coefficients import matrix_from_mask


def test_matrix_from_mask_rectangular():
    
    mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0 ,0, 0, 0, 0, 0]
    ], dtype=np.uint8)
    
    expected = np.array([
        [16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-4, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -4, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -4, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -4, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -4, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -4, 16, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-4, 0, 0, 0, 0, 0, 0, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -4, 0, 0, 0, 0, 0, -4, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -4, 0, 0, 0, 0, 0, -4, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, -4, 0, 0, 0, 0, 0, -4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, -4, 0, 0, 0, 0, 0, -4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, 0, 0, 0, 0, 0, 0, -4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 16, -4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, -4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, -4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, -4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, -4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16, -4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, -4, 16]
    ])
    
    data, row, col, height, width = matrix_from_mask(mask)
    m = sparse.csr_matrix((data, (row, col)), shape=(height, width)).toarray()
    assert np.all(expected == m)


# def test_matrix_from_mask():
#
#     mask = np.array([
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 1, 1, 0, 0, 0, 0],
#         [0, 1, 1, 1, 1, 1, 1, 0, 0],
#         [0, 0, 1, 1, 1, 1, 0, 0, 0],
#         [0, 0, 0, 1, 1, 1, 0, 0, 0],
#         [0, 0, 0, 0 ,0, 0, 0, 0, 0]
#     ], dtype=np.uint8)
#
#     expected = np.array([
#         [12, -4, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [-4, 14, -4, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, -4, 12, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 10, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [-4, 0, 0, -4, 16, -4, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0],
#         [0, -4, 0, 0, -4, 16, -4, 0, 0, 0, -4, 0, 0, 0, 0, 0],
#         [0, 0, -4, 0, 0, -4, 16, -4, 0, 0, 0, -4, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, -4, 14, -4, 0, 0, 0, -4, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, -4, 10, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, -4, 0, 0, 0, 0, 12, -4, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, -4, 0, 0, 0, -4, 16, -4, 0, -4, 0, 0],
#         [0, 0, 0, 0, 0, 0, -4, 0, 0, 0, -4, 16, -4, 0, -4, 0],
#         [0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, -4, 14, 0, 0, -4],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 12, -4, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, -4, 14, -4],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, -4, 12]
#        ])
#
#     data, row, col, height, width = matrix_from_mask(mask)
#
#     m = sparse.csr_matrix((data, (row, col)), shape=(height, width)).toarray()
#     assert np.all(expected == m)