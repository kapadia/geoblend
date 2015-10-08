
import benchmark
import numpy as np
import rasterio as rio
from skimage.morphology import disk
from geoblend.coefficients import matrix_from_mask


class Benchmark_Coefficients(benchmark.Benchmark):
    
    def setUp(self):
        self.mask200 = np.pad(disk(200), 2, mode='constant')
        self.mask400 = np.pad(disk(400), 2, mode='constant')
        self.mask800 = np.pad(disk(800), 2, mode='constant')
        self.mask1500 = np.pad(disk(1500), 2, mode='constant')

    def test_cython_disk_200(self):
        mat = matrix_from_mask(self.mask200)

    def test_cython_disk_400(self):
        mat = matrix_from_mask(self.mask400)

    def test_cython_disk_800(self):
        mat = matrix_from_mask(self.mask800)

    def test_cython_disk_1500(self):
        mat = matrix_from_mask(self.mask1500)


if __name__ == '__main__':
    benchmark.main(format='markdown', each=10)