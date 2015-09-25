
import click
import numpy as np
import rasterio as rio
from scipy import sparse

import pyamg
from pyamg.relaxation.smoothing import change_smoothers

from geoblend import blend
from geoblend.coefficients import matrix_from_mask
from geoblend.utilities import get_mask
from geoblend.solver import create_multilevel_solver


@click.command('geoblend')
@click.argument('srcpath', type=click.Path(exists=True))
@click.argument('refpath', type=click.Path(exists=True))
@click.argument('dstpath', type=click.Path(exists=False))
@click.option("--matrix")
@click.option("--gradient-multiplier", default=1.0, type=click.FLOAT)
def geoblend(srcpath, refpath, dstpath, matrix, gradient_multiplier):
    """
    Poisson blend the source image against the reference image.
    """

    mask = get_mask(srcpath)
    indices = np.nonzero(mask)

    with rio.drivers():
        with rio.open(srcpath) as src, rio.open(refpath) as ref:

            profile = src.profile

            if matrix:
                levels = load_multilevel_solver(matrix)
                ml = pyamg.multilevel.multilevel_solver(levels, coarse_solver='pinv2')
                change_smoothers(ml, 'gauss_seidel', 'gauss_seidel')
            else:
                mat = matrix_from_mask(mask)

                v = np.ones((mat.shape[0], 1))
                ml = pyamg.smoothed_aggregation_solver(mat, v, max_coarse=10)

            shape = src.shape

            with rio.open(dstpath, 'w', **profile) as dst:

                for bidx in range(1, 4):

                    source = src.read(bidx)
                    reference = ref.read(bidx)

                    arr = blend(source, reference, mask, ml, gradient_multiplier)
                    dst.write_band(bidx, arr)


if __name__ == '__main__':
    geoblend()
