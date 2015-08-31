
import benchmark
import numpy as np
from skimage.morphology import disk
from geoblend.coefficients import matrix_from_mask as matrix_from_mask_cython
from geoblend import matrix_from_mask_numba


class Benchmark_Coefficients(benchmark.Benchmark):
    
    def setUp(self):
        self.mask = np.pad(disk(2000), 2, mode='constant')
    
    def test_cython(self):
        mat = matrix_from_mask_cython(self.mask)
    
    def test_numba(self):
        mat = matrix_from_mask_numba(self.mask)



if __name__ == '__main__':
    benchmark.main(format='markdown')