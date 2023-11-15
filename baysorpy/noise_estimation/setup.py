#from distutils.core import setup, Extension
#from Cython.Build import cythonize
#import numpy

#package = Extension('noise_estimation', ['noise_estimation.pyx'], include_dirs=[numpy.get_include()])
#setup(ext_modules=cythonize([package]))

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("noise_estimation",       ["noise_estimation.pyx"]),
]

setup(
  name = 'BaysorCython',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)