
import os
import benchmark
import numpy as np
import rasterio as rio
from skimage.morphology import disk

from geoblend.vector import create_vector_slow
from geoblend import create_vector_numba
from geoblend.vector_fast import create_vector_fast as create_vector_cython


class Benchmark_Coefficients(benchmark.Benchmark):

    def setUp(self):

        directory = os.path.dirname(os.path.realpath(__file__))
        fixtures = os.path.join(directory, 'tests', 'fixtures')

        srcpath = os.path.join(fixtures, 'source.tif')
        refpath = os.path.join(fixtures, 'reference.tif')

        with rio.open(srcpath) as src:
            self.source = src.read(1)
        with rio.open(refpath) as ref:
            self.reference = ref.read(1)
        
        # Create a non-rectangular mask
        d = disk(60)
        dim = 121   # disk dimension is 121x121
        d2 = 0.5 * dim
        
        height, width = source.shape
        h2, w2 = 0.5 * height, 0.5 * width
        
        y0, y1 = int(h2 - d2), int(h2 + d2)
        x0, x1 = int(w2 - d2), int(w2 + d2)
        
        self.mask = np.zeros_like(self.source)
        self.mask[y0:y1, x0:x1] = d

    def test_cython(self):
        vector = create_vector_cython(self.mask, self.source, self.reference)

    def test_numba(self):
        vector = create_vector_numba(self.mask, self.source, self.reference)

    def test_python(self):
        vector = create_vector_slow(self.mask, self.source, self.reference)


if __name__ == '__main__':
    benchmark.main(format='markdown')