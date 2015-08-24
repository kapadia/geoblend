
import click
import numpy as np
from scipy import sparse
from scipy.ndimage import convolve

import pyamg
from pyamg.relaxation.smoothing import change_smoothers
import rasterio as rio
from skimage.morphology import binary_erosion, square

import geoblend
from geoblend.coefficients import matrix_from_mask
from geoblend.coefficient_matrix import create_multilevel_solver
from geoblend.b import b
from geoblend.blend import load_levels


@click.group()
def geoblend():
    pass


def create_vector(mask, source, reference):

    indices = np.nonzero(mask)
    
    selem = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    m = convolve(mask.astype(np.float), selem)
    m[m < 0] = 0
    m[m > 0] = 1
    bindices = np.nonzero(m)

    shape = source.shape
    operator = pyamg.gallery.poisson(shape)
    field = (operator * source.ravel()).reshape(shape)
    field[bindices] = reference[bindices]

    return field[indices]


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
            
            # For now the mask should be boolean
            mask = src.read(4)
            indices = np.nonzero(mask)
            mask[indices] = 1
            mask = mask.astype(np.uint8)
            selem = square(4)
            mask = binary_erosion(mask, selem).astype(np.uint8)
            indices = np.nonzero(mask)
            
            if matrix:
                levels = load_levels(matrix)
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
                    
                    field = (operator * source.ravel()).reshape(shape).astype(np.int32)
                    vector = create_vector(mask, source, reference).astype('float64')

                    x0 = source[indices].astype('float64')
                    pixels = np.round(ml.solve(b=vector, x0=x0, tol=1e-16))
                    np.clip(pixels, 0, 4095, pixels)

                    reference[indices] = pixels

                    dst.write_band(bidx, reference)


@click.command('create-solver')
@click.argument('srcpath', type=click.Path(exists=True))
@click.argument('dstpath', type=click.Path(exists=False))
def create_solver(srcpath, dstpath):
    """
    Create a multi-level solver and cache to disk.
    """
    
    with rio.drivers():
        with rio.open(srcpath) as src:

            mask = src.read(4)
            indices = np.nonzero(mask)
            mask[indices] = 1
            mask = mask.astype(np.uint8)
            
            print mask
            selem = square(4)
            mask = binary_erosion(mask, selem).astype(np.uint8)
            
            print mask
            
            data, row, col, height, width = matrix_from_mask(mask)
            mat = sparse.csr_matrix((data, (row, col)), shape=(height, width))

    create_multilevel_solver(dstpath, mat)

geoblend.add_command(poisson)
geoblend.add_command(create_solver)


if __name__ == '__main__':
    geoblend()
