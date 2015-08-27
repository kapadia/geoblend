
import benchmark
import numpy as np
from skimage.morphology import disk
from geoblend.coeffs import matrix_from_mask_slow
from geoblend.coeffs_fast import matrix_from_mask_fast
from geoblend.coeffs_numba import matrix_from_mask_numba


# class Benchmark_Coefficients_Python(benchmark.Benchmark):
#
#     def setUp(self):
#         self.mask = np.pad(disk(50), 2, mode='constant')
#
#     def test_coefficients(self):
#         mat = matrix_from_mask_slow(self.mask)


class Benchmark_Coefficients(benchmark.Benchmark):
    
    def setUp(self):
        self.mask = np.pad(disk(200), 2, mode='constant')
    
    def test_cython(self):
        mat = matrix_from_mask_fast(self.mask)
    
    def test_numba(self):
        mat = matrix_from_mask_numba(self.mask)
    
    # def test_python(self):
    #     mat = matrix_from_mask_slow(self.mask)



if __name__ == '__main__':
    benchmark.main(format='markdown')