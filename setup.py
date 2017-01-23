
import os
import shutil
from codecs import open as codecs_open
import numpy as np
from setuptools import setup, find_packages
from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext
from distutils import errors
from Cython.Build import cythonize
from Cython.Compiler.Errors import CompileError


def check_for_openmp():
    """
    There does not seem to be a cross platform and standard way to check for
    OpenMP support. Attempt to compile a test script. Proceed with OpenMP
    implementation if it works.
    """

    distribution = Distribution()
    ext_options = {
        'extra_compile_args': ['-fopenmp'],
        'extra_link_args': ['-fopenmp']
    }

    extensions = [
        Extension('geoblend.openmp_check', ['geoblend/openmp_check.pyx'], **ext_options)
    ]

    build_extension = build_ext(distribution)
    build_extension.finalize_options()
    build_extension.extensions = cythonize(extensions, force=True)
    build_extension.run()


ext_options = {
    'include_dirs': [ np.get_include() ]
}
extensions = [
    Extension('geoblend.vector', ['geoblend/vector.pyx'], **ext_options),
    Extension('geoblend.convolve', ['geoblend/convolve.pyx'], **ext_options)
]

pkg_dir = os.path.dirname(os.path.realpath(__file__))
dst = os.path.join(pkg_dir, 'geoblend', 'coefficients.pyx')

try:
    check_for_openmp()
    ext_options['extra_compile_args'] = ['-fopenmp']
    ext_options['extra_link_args'] = ['-fopenmp']
    src = os.path.join(pkg_dir, 'geoblend', '_coefficients_omp.pyx')
except (errors.LinkError, errors.CompileError, CompileError):
    src = os.path.join(pkg_dir, 'geoblend', '_coefficients.pyx')

shutil.copy(src, dst)

extensions.append(
    Extension('geoblend.coefficients', ['geoblend/coefficients.pyx'], **ext_options),
)


# Get the long description from the relevant file
with codecs_open('README.rst', encoding='utf-8') as f:
    long_description = f.read()


setup(name='geoblend',
      version='0.2.3',
      description=u"Geo-aware poisson blending.",
      long_description=long_description,
      classifiers=[],
      keywords='',
      author=u"Amit Kapadia",
      author_email='amit@planet.com',
      url='https://github.com/kapadia/geoblend',
      license='MIT',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      ext_modules=cythonize(extensions),
      zip_safe=False,
      install_requires=[
          'click',
          # 'rasterio',
          'pyamg',
          'scipy',
          'scikit-image'
      ],
      extras_require={
          'test': ['pytest'],
          'development': [
              'cython>=0.23.0',
              'benchmark'
          ]
      },
      entry_points="""
      [console_scripts]
      geoblend=geoblend.scripts.cli:geoblend
      """
      )