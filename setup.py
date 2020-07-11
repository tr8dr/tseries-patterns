#!/usr/bin/env python
# distutils: language=c++"

from setuptools import find_packages
from distutils.core import setup, Extension
from Cython.Build import cythonize

import io
import os
import sys
import fnmatch
import numpy

try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit("Cython not found. Cython is needed to build this library.")

def find_cython_files (basedir='tseries_patterns'):
    matches = []
    for path, sub, files in os.walk(basedir):
        for filename in fnmatch.filter(files, '*.pyx'):
            matches.append (os.path.join (path, filename))

    return matches

def requirements(filename):
    reqs = list()
    with io.open(filename, encoding='utf-8') as f:
        for line in f.readlines():
            reqs.append(line.strip())
    return reqs

modules = cythonize(find_cython_files(), language='c++')

setup(
    name='tseries_patterns',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/tr8dr/tseries-patterns',
    license='MIT License',
    author='Jonathan Shore',
    author_email='jonathan.shore@gmail.com',
    description='Momentum / Trend labeling for financial timeseries',
    install_requires=requirements(filename="requirements.txt"),
    download_url = 'https://github.com/tr8dr/tseries-patterns/archive/V0_1.tar.gz',
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries",
    ],
    keywords=', '.join([
        "trend detection", "financial trends", "quant", "stocks", "trading", 
    ]),
    python_requires='>=3',
    project_urls={
        'Bug Reports': 'https://github.com/tr8dr/tseries-patterns/issues',
        'Source': 'https://github.com/tr8dr/tseries-patterns',
        'Documentation': 'https://github.com/tr8dr/tseries-patterns'
    },
    
    ext_modules=modules,
    include_dirs=[numpy.get_include()]
)
