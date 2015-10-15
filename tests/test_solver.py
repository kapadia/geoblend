
import os
import unittest
import numpy as np
import pyamg

from geoblend import solver


TEST_DIR = os.path.dirname(os.path.realpath(__file__))


class SolverTest(unittest.TestCase):


    def setUp(self):

        A = pyamg.gallery.poisson((100, 100), format='csr')
        self.ml = pyamg.aggregation.smoothed_aggregation_solver(A, B=None)


    def tearDown(self):
        
        fpath = os.path.join(TEST_DIR, 'ml.npz')
        os.unlink(fpath)


    def test_save(self):

        fpath = os.path.join(TEST_DIR, 'ml.npz')
        solver.save(fpath, self.ml)
        
        assert os.path.exists(fpath)


    def test_load(self):

        fpath = os.path.join(TEST_DIR, 'ml.npz')
        solver.save(fpath, self.ml)

        ml = solver.load(fpath)

        assert repr(self.ml) == repr(ml)
        