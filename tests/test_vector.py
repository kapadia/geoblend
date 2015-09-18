
import os
import numpy as np
from geoblend.vector import create_vector

TEST_DIR = os.path.dirname(os.path.realpath(__file__))


def test_vector_from_rectangular_mask():

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
    ], dtype=np.float64)

    reference = np.array([
        [611, 573, 639, 564, 626, 588, 556, 503, 458, 461],
        [689, 559, 532, 550, 572, 601, 521, 466, 469, 437],
        [631, 530, 513, 504, 545, 516, 428, 444, 447, 430],
        [648, 566, 518, 514, 592, 537, 518, 468, 658, 559],
        [553, 587, 556, 544, 423, 574, 546, 452, 456, 387],
        [590, 598, 583, 564, 408, 389, 219, 498, 501, 479],
        [565, 572, 564, 436, 442, 638, 208, 382, 466, 455],
        [566, 545, 570, 507, 429, 378, 425, 474, 425, 466]
    ], dtype=np.float64)

    expected = np.array([
        2844., 704., 1508., 886., 1280., 1928., 788., 1294., 
        708., -68., -1120., 656., -244., -96., -520., 812., 
        1226., 456., -144., 192., 184., 368., -292., 
        2092., 1208., 164., 208., -792., 800., 1128., 
        -1252., 1734., 926., 156., 772., -492., -408., 
        -2148., -1068., 3408., 2348., 1412., -40., 422., 
        2300., 744., 896., 1018.
    ])

    assert np.all(expected == create_vector(source, reference, mask))


def test_vector_from_mask():

    mask = np.array([
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
      [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
      [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
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
    ], dtype=np.float64)

    reference = np.array([
        [611, 573, 639, 564, 626, 588, 556, 503, 458, 461],
        [689, 559, 532, 550, 572, 601, 521, 466, 469, 437],
        [631, 530, 513, 504, 545, 516, 428, 444, 447, 430],
        [648, 566, 518, 514, 592, 537, 518, 468, 658, 559],
        [553, 587, 556, 544, 423, 574, 546, 452, 456, 387],
        [590, 598, 583, 564, 408, 389, 219, 498, 501, 479],
        [565, 572, 564, 436, 442, 638, 208, 382, 466, 455],
        [566, 545, 570, 507, 429, 378, 425, 474, 425, 466]
    ], dtype=np.float64)

    expected = np.array([
        1958., 2636., 1004., -1120., 2742., 2672., -144., 
        192., 1178., 1182., 2092., 1374., -792., 800., 
        1128., -1252., 4088., 2656., -492., -408., -1516., 
        1312., 1300., 2478.
    ])

    assert np.all(expected == create_vector(source, reference, mask))


def test_problem_image():
    mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)
   
    source = np.array([
        [231, 199, 236, 235, 242, 243,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [266, 223, 236, 244, 248, 243, 227,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [221, 195, 230, 243, 235, 220, 221, 221,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [214, 200, 226, 228, 232, 228, 226, 217, 229,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [210, 199, 215, 213, 229, 231, 218, 208, 221, 235,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [227, 220, 227, 219, 207, 222, 208, 224, 210, 215, 262,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [224, 215, 218, 212, 202, 194, 216, 207, 209, 249, 232, 252,   0,   0,   0,   0,   0,   0,   0,   0],
        [226, 227, 221, 201, 189, 206, 208, 225, 236, 222, 255, 244, 222,   0,   0,   0,   0,   0,   0,   0],
        [222, 217, 209, 202, 196, 206, 227, 222, 233, 226, 250, 217, 245, 240,   0,   0,   0,   0,   0,   0],
        [211, 212, 210, 203, 199, 220, 218, 218, 230, 258, 248, 234, 233, 251, 273,   0,   0,   0,   0,   0]
    ], dtype=np.float64)

    reference = np.array([
        [277, 277, 277, 270, 270, 270, 270, 270, 270, 262, 262, 262, 262, 262, 262, 286, 286, 286, 286, 286],
        [288, 288, 288, 274, 274, 274, 270, 270, 270, 262, 262, 262, 262, 262, 262, 286, 286, 286, 286, 286],
        [288, 288, 288, 274, 274, 274, 274, 274, 274, 251, 251, 251, 251, 251, 251, 281, 281, 281, 281, 281],
        [288, 288, 288, 274, 274, 274, 274, 274, 274, 251, 251, 251, 251, 251, 281, 281, 281, 281, 281, 281],
        [288, 288, 288, 274, 274, 274, 274, 274, 274, 251, 251, 251, 251, 251, 281, 281, 281, 281, 281, 281],
        [288, 288, 288, 274, 274, 274, 274, 274, 274, 251, 251, 251, 251, 251, 281, 281, 281, 281, 281, 281],
        [288, 288, 288, 274, 274, 274, 274, 274, 274, 251, 251, 251, 251, 251, 281, 281, 281, 281, 281, 281],
        [290, 290, 290, 271, 271, 271, 271, 271, 271, 272, 272, 272, 272, 272, 279, 279, 279, 279, 279, 279],
        [290, 290, 290, 271, 271, 271, 271, 271, 271, 272, 272, 272, 272, 272, 279, 279, 279, 279, 279, 279],
        [290, 290, 290, 271, 271, 271, 271, 271, 271, 272, 272, 272, 272, 272, 279, 279, 279, 279, 279, 279]
    ], dtype=np.float64)
    
    expected = np.array([1152., 598., 578., 1178., 252., 80., 140., -12., 406., 1060., 468., 124., -8., 32., 12., 80., 1064., 402., -20., -156., 132., 108., -4., -192., 572., 2080., 638., 144., 68., -176., 192., -192., 252., -116., -384., 2360., 478., -12., 32., 24., -280., 188., -184., -264., 472., -360., 2124., 694., 116., -80., -196., 108., -168., 108., 220., -312., 288., 120., 1754., 1152., 526., 540., 500., 470., 740., 474., 608., 372., 756., 158., 792., 2564.])
    
    assert np.all(expected == create_vector(source, reference, mask))