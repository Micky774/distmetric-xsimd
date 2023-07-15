# Scikit-Learn SIMD DistanceMetrics (SLSDM)

## Install From Source:

1. Create a new conda environment with `xsimd`: `conda create -n <env_name> -c conda-forge python~=3.10 xsimd`
2. Activate the conda environment: `conda activate <env_name>`
3. Run `PIP_EXTRA_INDEX_URL=https://pypi.anaconda.org/scipy-wheels-nightly/simple pip install -e .`

Note: if you are building with a custom development installation of scikit-learn (e.g. off a branch rather than `main`) then use the `--no-build-isolation`
flag to ensure it is not superceded by the nightly version.

## Specify SIMD Target Architectures

Coming soon.
