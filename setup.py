#!/usr/bin/env python

import runpy
from setuptools import setup, find_namespace_packages


SETUP = dict(
    name='liveml',
    version=runpy.run_path('liveml/version.py')['__version__'],
    description='Live Machine Learning Toolkit',
    long_description="",
    packages=find_namespace_packages(where='.', include=('liveml*', 'liveml.*')),
    install_requires=[
        'numpy',
    ],
    include_package_data=True
)

if __name__ == '__main__':
    setup(**SETUP)
