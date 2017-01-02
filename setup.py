from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

fast_opts = ["-mtune=native", "-march=native", "-O3", 
                "-ftree-vectorize", "-msse2", "-msse3", "-fPIC", "-ffast-math", 
                "-std=c99", "-msse", "-mfpmath=sse", "-Wno-unused-function"]

setup(
    cmdclass = {'build_ext': build_ext},
    include_dirs = [np.get_include()],
    ext_modules = [
		Extension("averaged_perceptron", ["averaged_perceptron.pyx"],
            include_dirs=[np.get_include()],
		    extra_compile_args=fast_opts)
    ]
)

