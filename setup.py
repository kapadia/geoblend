
import os
from codecs import open as codecs_open
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

# TODO: Generalize this. Check that openmp is available.
os.environ["CC"] = "/usr/local/Cellar/gcc/5.1.0/bin/gcc-5"

# Get the long description from the relevant file
with codecs_open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

ext_options = {
    "include_dirs": [ np.get_include() ],
    "extra_compile_args": [ '-fopenmp' ],
    "extra_link_args": [ '-fopenmp' ]
}

extensions = [
    Extension('geoblend.coefficients', ['geoblend/coefficients.pyx'], **ext_options),
    Extension('geoblend.vector', ['geoblend/vector.pyx'], **ext_options)
]

setup(name='geoblend',
      version='0.1.0',
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
          'rasterio',
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