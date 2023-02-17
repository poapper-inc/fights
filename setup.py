from os.path import join

import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

fights_envs_path = join("fights", "envs")
defs = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
puoribor = Extension(
    "puoribor_cython",
    sources=[join(fights_envs_path, "puoribor_cython.pyx")],
    include_dirs=[np.get_include()],
    define_macros=defs,
)
quoridor = Extension(
    "quoridor_cython",
    sources=[join(fights_envs_path, "quoridor_cython.pyx")],
    include_dirs=[np.get_include()],
    define_macros=defs,
)

setup(ext_modules=cythonize([puoribor, quoridor]))
