#!/usr/bin/env python

from distutils.core import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="SynRD",
    version="0.1",
    description="Synthetic Datasets generator tester",
    long_description=long_description,
    author="Lucas Rosenblatt",
    author_email="lr2872@nyu.edu",
    url="https://github.com/DataResponsibly/SynRD",
    packages=["SynRD"],
    # setup_requires=['wheel'],
    install_requires=["smartnoise-synth", "rpy2"],
)
