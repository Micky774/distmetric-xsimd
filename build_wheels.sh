#!/bin/bash

git clone -c feature.manyFiles=true https://github.com/spack/spack.git
spack/bin/spack install xsimd
spack/bin/spack load xsimd

which python
python -m pip install cibuildwheel==2.13.1

export CIBW_BUILD="cp39*"
python -m cibuildwheel --output-dir wheelhouse
