
import benchmark
import numpy as np
from skimage.morphology import disk
from geoblend.coefficients import matrix_from_mask


class Benchmark_Coefficients(benchmark.Benchmark):
    
    def setUp(self):
        self.mask = np.pad(disk(200), 2, mode='constant')
    
    def test_cython(self):
        mat = matrix_from_mask(self.mask)



if __name__ == '__main__':
    benchmark.main(format='markdown')