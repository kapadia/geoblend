
import benchmark
import numpy as np
from skimage.morphology import disk
from geoblend.coefficients import matrix_from_mask


class Benchmark_Coefficients(benchmark.Benchmark):

    def test_cython_disk_200(self):
        mask = np.pad(disk(200), 2, mode='constant')
        mat = matrix_from_mask(mask)

    def test_cython_disk_400(self):
        mask = np.pad(disk(400), 2, mode='constant')
        mat = matrix_from_mask(mask)

    def test_cython_disk_600(self):
        mask = np.pad(disk(600), 2, mode='constant')
        mat = matrix_from_mask(mask)


if __name__ == '__main__':
    benchmark.main(format='markdown')