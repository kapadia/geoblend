
import numpy as np
from scipy import sparse
import pyamg

from geoblend.coefficient_matrix import create_coefficient_matrix
from geoblend.coefficients import matrix_from_mask
from geoblend.b import b


def test_blend_rectangular():
    
    mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0 ,0, 0, 0, 0, 0]
    ], dtype=np.uint8)

    source = np.array([
        [459, 516, 579, 595, 622, 575, 564, 607, 627],
        [451, 509, 540, 561, 528, 515, 537, 621, 614],
        [489, 537, 574, 587, 497, 484, 491, 540, 532],
        [543, 683, 650, 620, 546, 513, 521, 522, 549],
        [557, 656, 666, 628, 576, 570, 571, 545, 546],
        [623, 742, 683, 610, 616, 566, 558, 516, 501]
    ], dtype=np.uint16)

    reference = np.array([
        [555, 555, 514, 514, 514, 514, 479, 479, 479],
        [555, 555, 514, 514, 514, 514, 479, 479, 479],
        [555, 555, 514, 514, 514, 514, 479, 479, 479],
        [504, 504, 472, 472, 472, 472, 458, 458, 458],
        [504, 504, 472, 472, 472, 472, 458, 458, 458],
        [504, 504, 472, 472, 472, 472, 458, 458, 458]
    ], dtype=np.uint16)

    field = np.array([
        [869,  517,  665,  618,  790,  599,  537,  616, 1287],
        [347,   -8,  -63,   -6,  -83,  -64,  -43,  186,  676],
        [425, -107,  -18,   96, -157,  -80, -118,   -6,  425],
        [443,  346,   57,   69,  -22,  -69,  -13,  -67,  596],
        [406,  -24,   47,   40,  -56,   54,   90,   25,  589],
        [1193, 1006,  714,  513,  712,  520,  579,  460,  942]
    ], dtype=np.int32)

    expected = np.array([
        [555, 555, 514, 514, 514, 514, 479, 479, 479],
        [555, 533, 498, 489, 441, 433, 446, 511, 479],
        [555, 530, 521, 510, 409, 396, 402, 455, 479],
        [504, 619, 563, 524, 448, 419, 430, 435, 458],
        [504, 533, 530, 507, 460, 471, 478, 464, 458],
        [504, 504, 472, 472, 472, 472, 458, 458, 458]
    ], dtype=np.uint16)
    
    data, row, col, height, width = matrix_from_mask(mask)
    vector = b(mask, field, reference)
    mat = sparse.csr_matrix((data, (row, col)), shape=(height, width))
    
    indices = np.nonzero(mask)
    
    ml = pyamg.smoothed_aggregation_solver(mat, np.ones((mat.shape[0], 1)), max_coarse=10)
    x0 = np.ravel(source[indices]).astype('float64')
    
    pixels = np.round(ml.solve(b=vector, x0=x0, tol=1e-05))
    
    img = np.copy(reference)
    img[indices] = pixels
    
    # TODO: This test is a little arbitrary. The expected array is derived from
    #       sympy (analytic), whereas the img array is numerically computed. Find
    #       an appropriate way to compare these.
    n = float(expected.size)
    y0, y1 = img.min(), img.max()
    err = np.sqrt(np.power(expected.astype(np.float) - img.astype(np.float), 2).sum() / n) / (y1 - y0)
    
    assert err < 0.1
    
def test_blend():
    
    mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)

    source = np.array([
        [459, 516, 579, 595, 622, 575, 564, 607, 627],
        [451, 509, 540, 561, 528, 515, 537, 621, 614],
        [489, 537, 574, 587, 497, 484, 491, 540, 532],
        [543, 683, 650, 620, 546, 513, 521, 522, 549],
        [557, 656, 666, 628, 576, 570, 571, 545, 546],
        [623, 742, 683, 610, 616, 566, 558, 516, 501]
    ], dtype=np.uint16)

    reference = np.array([
        [555, 555, 514, 514, 514, 514, 479, 479, 479],
        [555, 555, 514, 514, 514, 514, 479, 479, 479],
        [555, 555, 514, 514, 514, 514, 479, 479, 479],
        [504, 504, 472, 472, 472, 472, 458, 458, 458],
        [504, 504, 472, 472, 472, 472, 458, 458, 458],
        [504, 504, 472, 472, 472, 472, 458, 458, 458]
    ], dtype=np.uint16)

    field = np.array([
        [869,  517,  665,  618,  790,  599,  537,  616, 1287],
        [347,   -8,  -63,   -6,  -83,  -64,  -43,  186,  676],
        [425, -107,  -18,   96, -157,  -80, -118,   -6,  425],
        [443,  346,   57,   69,  -22,  -69,  -13,  -67,  596],
        [406,  -24,   47,   40,  -56,   54,   90,   25,  589],
        [1193, 1006,  714,  513,  712,  520,  579,  460,  942]
    ], dtype=np.int32)

    expected = np.array([
        [555, 555, 514, 514, 514, 514, 479, 479, 479],
        [555, 533, 498, 489, 441, 433, 446, 511, 479],
        [555, 530, 521, 510, 409, 396, 402, 455, 479],
        [504, 619, 563, 524, 448, 419, 430, 435, 458],
        [504, 533, 530, 507, 460, 471, 478, 464, 458],
        [504, 504, 472, 472, 472, 472, 458, 458, 458]
    ], dtype=np.uint16)
    
    data, row, col, height, width = matrix_from_mask(mask)
    vector = b(mask, field, reference)
    mat = sparse.csr_matrix((data, (row, col)), shape=(height, width))
    
    indices = np.nonzero(mask)
    
    ml = pyamg.smoothed_aggregation_solver(mat, np.ones((mat.shape[0], 1)), max_coarse=10)
    x0 = np.ravel(source[indices]).astype('float64')
    
    pixels = np.round(ml.solve(b=vector, x0=x0, tol=1e-05))
    
    img = np.copy(reference)
    img[indices] = pixels
    
    # TODO: This test is a little arbitrary. The expected array is derived from
    #       sympy (analytic), whereas the img array is numerically computed. Find
    #       an appropriate way to compare these.
    #
    #       Also, the RMSE is a function of the sample size. It makes sense that the RMSE
    #       is larger for the non-rectangular mask since less pixels are computed.
    n = float(expected.size)
    y0, y1 = img.min(), img.max()
    err = np.sqrt(np.power(expected.astype(np.float) - img.astype(np.float), 2).sum() / n) / (y1 - y0)
    
    assert err < 0.16

