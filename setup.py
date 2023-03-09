from os.path import join

import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

fights_envs_path = join("src", "fights", "envs")
defs = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
puoribor = Extension(
    "fights.envs.puoribor_cython",
    sources=[join(fights_envs_path, "puoribor_cython.pyx")],
    include_dirs=[np.get_include()],
    define_macros=defs,
)
quoridor = Extension(
    "fights.envs.quoridor_cython",
    sources=[join(fights_envs_path, "quoridor_cython.pyx")],
    include_dirs=[np.get_include()],
    define_macros=defs,
)
othello = Extension(
    "fights.envs.othello_cythonfn",
    sources=[join(fights_envs_path, "othello_cythonfn.pyx")],
    include_dirs=[np.get_include()],
    define_macros=defs,
)

setup(ext_modules=cythonize([puoribor, quoridor, othello]))
