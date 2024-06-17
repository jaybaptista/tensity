#!/usr/bin/env python

from distutils.core import setup

setup(
    name="tensity",
    version="1.0",
    description="Tensorial Minkowski functionals on molecular clouds",
    author="Jay Baptista",
    author_email="jaymarie@stanford.edu",
    packages=["tensity"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "asdf",
        "h5py",
    ],
)