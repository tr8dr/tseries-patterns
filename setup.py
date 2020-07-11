from setuptools import setup, find_packages, Extension
import io

try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit("Cython not found. Cython is needed to build this library.")

def requirements(filename):
    reqs = list()
    with io.open(filename, encoding='utf-8') as f:
        for line in f.readlines():
            reqs.append(line.strip())
    return reqs


setup(
    name='trendet',
    version='0.7',
    packages=find_packages(),
    url='https://github.com/tr8dr/mvlabeler',
    license='MIT License',
    author='Jonathan Shore',
    author_email='jonathan.shore@gmail.com',
    description='Momentum / Trend labeling for financial timeseries',
    install_requires=requirements(filename="requirements.txt"),
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
    extras_require={
        "tests": requirements(filename='tests/requirements.txt'),
        "docs": requirements(filename='docs/requirements.txt')
    },
    project_urls={
        'Bug Reports': 'https://github.com/tr8dr/mvlabeler/issues',
        'Source': 'https://github.com/tr8dr/mvlabeler',
        'Documentation': 'https://github.com/tr8dr/mvlabeler'
    },
    
    ext_modules=cythonize('mvlabeler/TrendLabeler.pyx', compiler_directives={'embedsignature': True})
)
