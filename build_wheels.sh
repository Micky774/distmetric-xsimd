#!/bin/bash

source ~/.profile
mamba list
which python
python -m pip install cibuildwheel==2.13.1

export CIBW_BUILD="cp39*"
python -m cibuildwheel --output-dir wheelhouse
