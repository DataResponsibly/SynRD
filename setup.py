#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="SynRD",
    version="0.1",
    python_requires='== 3.7.*', # Unfortunately, need specific python version for rpy2
    description="Benchmark for differentially private synthetic data.",
    long_description=long_description,
    author="Lucas Rosenblatt",
    author_email="lr2872@nyu.edu",
    url="https://github.com/DataResponsibly/SynRD",
    packages=["SynRD", 
              "SynRD.synthesizers",
              "SynRD.benchmark",
              "SynRD.papers",
              "SynRD.datasets"],
    package_data={'SynRD': ['papers/process.R']},
    # setup_requires=['wheel'],
    install_requires=["DataSynthesizer",
                     "smartnoise-synth", 
                      "pandas", 
                      "numpy", 
                      "tqdm",
                      "dill",
                      "requests",
                      "scikit-learn",
                      "disjoint-set",
                      "networkx",
                      "diffprivlib", 
                      "pathlib",
                      "statsmodels"],
)

# NOTE: Independent installation of mbi required with:
# `pip install git+https://github.com/ryan112358/private-pgm.git` 

# NOTE: Independent installation of rpy2 required as well, see README.