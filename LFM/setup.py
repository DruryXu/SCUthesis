#!/usr/bin/env python
# coding=utf-8
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("gradient_descent.pyx"), include_dirs=[numpy.get_include()])
