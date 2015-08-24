
import click
import numpy as np
import rasterio as rio
from scipy import sparse

import pyamg
from pyamg.relaxation.smoothing import change_smoothers
from skimage.morphology import binary_erosion, square

from geoblend import blend
from geoblend.coefficients import matrix_from_mask
from geoblend.solver import create_multilevel_solver, load_multilevel_solver


def prepare_mask(mask):
    """
    Temporary function to prepare a mask band for experiments.
    """

    indices = np.nonzero(mask)
    mask[indices] = 1

    selem = square(4)
    return binary_erosion(mask, selem).astype(np.uint8)


@click.group()
def geoblend():
    pass


@click.command('poisson')
@click.argument('srcpath', type=click.Path(exists=True))
@click.argument('refpath', type=click.Path(exists=True))
@click.argument('dstpath', type=click.Path(exists=False))
@click.option("--matrix")
def poisson(srcpath, refpath, dstpath, matrix):
    """
    Poisson blend the source image against the reference image.
    
    .. todo:: Move logic into library function.
    """

    with rio.drivers():
        with rio.open(srcpath) as src, rio.open(refpath) as ref:
            
            profile = src.profile
            
            mask = src.read(4)
            mask = prepare_mask(mask)
            indices = np.nonzero(mask)
            
            if matrix:
                levels = load_multilevel_solver(matrix)
                ml = pyamg.multilevel.multilevel_solver(levels, coarse_solver='pinv2')
                change_smoothers(ml, 'gauss_seidel', 'gauss_seidel')
            else:
                data, row, col, height, width = matrix_from_mask(mask)
                mat = sparse.csr_matrix((data, (row, col)), shape=(height, width))

                v = np.ones((mat.shape[0], 1))
                ml = pyamg.smoothed_aggregation_solver(mat, v, max_coarse=10)

            shape = src.shape
            operator = pyamg.gallery.poisson(shape)

            with rio.open(dstpath, 'w', **profile) as dst:

                for bidx in range(1, 4):

                    source = src.read(bidx)
                    reference = ref.read(bidx)

                    arr = blend(source, reference, mask, operator, ml)
                    dst.write_band(bidx, arr)


@click.command('create-solver')
@click.argument('srcpath', type=click.Path(exists=True))
@click.argument('dstpath', type=click.Path(exists=False))
def create_solver(srcpath, dstpath):
    """
    Create a multi-level solver and save to disk.
    """

    with rio.drivers():
        with rio.open(srcpath) as src:

            mask = src.read(4)
            mask = prepare_mask(mask)
            indices = np.nonzero(mask)

            data, row, col, height, width = matrix_from_mask(mask)
            mat = sparse.csr_matrix((data, (row, col)), shape=(height, width))

    create_multilevel_solver(dstpath, mat)


geoblend.add_command(poisson)
geoblend.add_command(create_solver)


if __name__ == '__main__':
    geoblend()
