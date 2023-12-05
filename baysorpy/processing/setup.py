from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension(name = "noise_estimation",
              sources=["noise_estimation.pyx"],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp']),
    Extension(name = "molecule_clustering_cython",
              sources=["molecule_clustering_cython.pyx"],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp']),
]

setup(
  name = 'BaysorCython',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)