
import os
import numpy as np
from geoblend.convolve import convolve_mask_aware
from scipy.ndimage import convolve

TEST_DIR = os.path.dirname(os.path.realpath(__file__))


def test_convolve_mask_aware():
    
    selem = np.array([
        [  0, -1,  0 ],
        [ -1,  4, -1 ], 
        [  0, -1,  0 ]
    ], dtype=np.float64)

    mask = np.array([
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)

    source = np.array([
        [502, 527, 545, 517, 518, 492, 457, 562, 405, 420],
        [605, 512, 444, 473, 465, 496, 527, 445, 387, 397],
        [543, 446, 440, 393, 491, 472, 471, 417, 439, 371],
        [513, 476, 494, 448, 470, 491, 492, 443, 559, 514],
        [454, 487, 498, 471, 402, 484, 471, 377, 574, 452],
        [507, 478, 499, 484, 381, 372, 249, 333, 607, 410],
        [451, 497, 497, 392, 389, 476, 357, 366, 400, 464],
        [485, 517, 567, 531, 443, 324, 370, 408, 361, 464]
    ], dtype=np.uint16)
    
    expected = convolve(source.astype(np.float64), selem, mode='reflect')
    gradient = convolve_mask_aware(source, mask)

    # Should match except at the boundaries due to different rules
    assert np.all(expected[2:-2, 2:-2] == gradient[2:-2, 2:-2])


if __name__ == '__main__':
    test_convolve_mask_aware()