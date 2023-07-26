#! /usr/bin/env python
#
# Copyright (C) 2007-2009 Cournapeau David <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
# License: 3-clause BSD

import sys
import os
from os.path import join
import platform
import shutil

from setuptools import Command, Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

from sklearn._build_utils import _check_cython_version  # noqa
from sklearn.externals._packaging.version import parse as parse_version  # noqa
from slsdm._generate import generate_code, GENERATED_DIR, _parse_xsimd_to_arch
import traceback
import importlib
from collections import defaultdict
import contextlib
from slsdm import __version__

VERSION = __version__
SRC_NAME = "slsdm"
DISTNAME = "slsdm"
DESCRIPTION = "A set of SIMD-accelerated DistanceMetric implementations"
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "Meekail Zain"
MAINTAINER_EMAIL = "zainmeekail@gmail.com"
LICENSE = "new BSD"
XSIMD_ARCHS = None
METRICS = None

PYTEST_MIN_VERSION = "5.4.3"
CYTHON_MIN_VERSION = "0.29.33"
SKLEARN_MIN_VERSION = "1.3.0"

# TODO: Enable and trim as needed
# 'build' and 'install' is included to have structured metadata for CI.
# It will NOT be included in setup's extras_require
# The values are (version_spec, comma separated tags)
dependent_packages = {
    "scikit-learn": (SKLEARN_MIN_VERSION, "install"),
    "cython": (CYTHON_MIN_VERSION, "build"),
    "pytest": (PYTEST_MIN_VERSION, "tests"),
    "pytest-cov": ("2.9.0", "tests"),
    "flake8": ("3.8.2", "tests"),
    "black": ("23.3.0", "tests"),
    "mypy": ("0.961", "tests"),
}


# create inverse mapping for setuptools
tag_to_packages: dict = defaultdict(list)
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        tag_to_packages[extra].append("{}>={}".format(package, min_version))


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        if self.parallel is None:
            # Do not override self.parallel if already defined by
            # command-line flag (--parallel or -j)

            parallel = os.environ.get("SLSDM_BUILD_PARALLEL")
            if parallel:
                self.parallel = int(parallel)
        if self.parallel:
            print("setting parallel=%d " % self.parallel)

    def run(self):
        # Specifying `build_clib` allows running `python setup.py develop`
        # fully from a fresh clone.
        self.run_command("build_clib")
        _build_ext.run(self)


# Custom clean command to remove build artifacts
class CleanCommand(Command):
    description = "Remove build artifacts from the source tree"

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, "PKG-INFO"))
        if remove_c_files:
            print("Will remove generated .c files")
        if os.path.exists("build"):
            shutil.rmtree("build")
        for dirpath, dirnames, filenames in os.walk(SRC_NAME):
            for filename in filenames:
                root, extension = os.path.splitext(filename)

                if extension in [".so", ".pyd", ".dll", ".pyc"]:
                    os.unlink(os.path.join(dirpath, filename))

                if remove_c_files and extension in [".c", ".cpp"] and "src" not in root:
                    pyx_file = str.replace(filename, extension, ".pyx")
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))

                if remove_c_files and extension == ".tp":
                    if os.path.exists(os.path.join(dirpath, root)):
                        os.unlink(os.path.join(dirpath, root))

            for dirname in dirnames:
                if dirname in ("__pycache__", "generated"):
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {
    "clean": CleanCommand,
    "build_ext": build_ext,
}


def check_package_status(package, min_version):
    """
    Returns a dictionary containing a boolean specifying whether given package
    is up-to-date, along with the version string (empty string if
    not installed).
    """
    package_status = {}
    try:
        module = importlib.import_module(package)
        package_version = module.__version__
        package_status["up_to_date"] = parse_version(package_version) >= parse_version(
            min_version
        )
        package_status["version"] = package_version
    except ImportError:
        traceback.print_exc()
        package_status["up_to_date"] = False
        package_status["version"] = ""

    req_str = "SLSDM requires {} >= {}.\n".format(package, min_version)

    if package_status["up_to_date"] is False:
        if package_status["version"]:
            raise ImportError(
                "Your installation of {} {} is out-of-date.\n{}".format(
                    package, package_status["version"], req_str
                )
            )
        else:
            raise ImportError("{} is not installed.\n{}".format(package, req_str))


def _parse_arch_to_flag(arch, compiler="gcc"):
    REQUIRED_FLAGS = {
        "avx512bw": ["avx512f", "avx512cd", "avx512dq"],
        "avx512dq": ["avx512f", "avx512cd"],
        "avx512cd": ["avx512f"],
        "avx512f": [],
        "fma4": [],
        "avx2": [],
        "avx": [],
        "sse4.2": [],
        "sse4.1": [],
        "ssse3": [],
        "sse3": [],
        "sse2": [],
    }
    flags = []
    if compiler == "gcc":
        if "fma3" in arch:
            flags.append("-mfma")
            arch = arch[5:]
        arch = arch.replace("_", ".")
        for flag in (arch, *REQUIRED_FLAGS[arch]):
            flags.append("-m" + flag)
    else:
        raise ValueError("Only gcc is currently supported.")
    return flags


def _make_library_config(metrics, xsimd_archs):
    libraries = []
    for arch in xsimd_archs:
        arch = _parse_xsimd_to_arch(arch)
        flags = _parse_arch_to_flag(arch)
        libraries.append(
            (
                arch,
                {
                    "language": "c++",
                    "sources": [f"{join(GENERATED_DIR, arch)}.cpp"],
                    "depends": [
                        join(GENERATED_DIR, f"{metric}.hpp") for metric in metrics
                    ],
                    "cflags": ["-std=c++14", *flags],
                    "extra_link_args": ["-std=c++14"],
                    "include_dirs": ["slsdm/src/", "xsimd/include/"],
                },
            )
        )
    return libraries


def build_extension_config():
    # TODO: Filter to only delete unsupported architectures and files that have
    # had their corresponding *.def files altered.
    if os.path.exists(GENERATED_DIR):
        shutil.rmtree(GENERATED_DIR)

    # TODO: Filter to only generate files that are either missing or have had
    # their corresponding *.def files altered.
    # Generate simd compilation targets from *.def files
    target_arch = os.environ.get("SLSDM_SIMD_ARCH", "<=avx")
    metrics, xsimd_archs = generate_code(target_arch)
    srcs = [
        "_dist_metrics.pyx.tp",
        "_dist_metrics.pxd",
        join(GENERATED_DIR[6:], "_dist_optim.cpp"),
    ]
    extension_config = {
        SRC_NAME: [
            {
                "sources": srcs,
                "language": "c++",
                "include_dirs": ["src/", "../xsimd/include/"],
                "extra_compile_args": ["-std=c++14"],
                "extra_link_args": ["-std=c++14"],
            },
        ],
    }
    return extension_config, metrics, xsimd_archs


def cythonize_extensions(extension):
    """Check that a recent Cython is available and cythonize extensions"""
    _check_cython_version()
    from Cython.Build import cythonize

    n_jobs = 1
    with contextlib.suppress(ImportError):
        import joblib

        n_jobs = joblib.cpu_count()

    # Additional checks for Cython
    cython_enable_debug_directives = (
        os.environ.get("SLSDM_ENABLE_DEBUG_CYTHON_DIRECTIVES", "0") != "0"
    )

    compiler_directives = {
        "language_level": 3,
        "boundscheck": cython_enable_debug_directives,
        "wraparound": False,
        "initializedcheck": False,
        "nonecheck": False,
        "cdivision": True,
    }

    return cythonize(
        extension,
        nthreads=n_jobs,
        force=True,
        compiler_directives=compiler_directives,
    )


def configure_extension_modules():
    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    if "sdist" in sys.argv or "--help" in sys.argv:
        return [], None, None

    # Always use NumPy 1.7 C API for all compiled extensions.
    # See: https://numpy.org/doc/stable/reference/c-api/deprecations.html
    DEFINE_MACRO_NUMPY_C_API = (
        "NPY_NO_DEPRECATED_API",
        "NPY_1_7_API_VERSION",
    )
    from sklearn._build_utils import gen_from_templates
    import numpy

    is_pypy = platform.python_implementation() == "PyPy"
    np_include = numpy.get_include()
    default_optimization_level = "O3"

    if os.name == "posix":
        default_libraries = ["m"]
    else:
        default_libraries = []

    # Necessary to generate necessary SIMD instructions
    # TODO: Update to compile each compilation unit separately using the
    # required flags, in order to avoid accidental vector promotion
    # (e.g SSE3->AVX)
    march_flag = os.environ.get("SLSDM_MARCH", None)
    default_extra_compile_args = (
        [f"-march={march_flag}"] if march_flag is not None else []
    )
    build_with_debug_symbols = os.environ.get("SLSDM_ENABLE_DEBUG_SYMBOLS", "0") != "0"
    if os.name == "posix":
        if build_with_debug_symbols:
            default_extra_compile_args.append("-g")
        else:
            # Setting -g0 will strip symbols, reducing the binary size of extensions
            default_extra_compile_args.append("-g0")

    cython_exts = []
    extension_config, metrics, xsimd_archs = build_extension_config()
    for submodule, extensions in extension_config.items():
        submodule_parts = submodule.split(".")
        parent_dir = join(*submodule_parts)
        for extension in extensions:
            if is_pypy and not extension.get("compile_for_pypy", True):
                continue

            # Generate files with Tempita
            tempita_sources = []
            sources = []
            for source in extension["sources"]:
                source = join(parent_dir, source)
                new_source_path, path_ext = os.path.splitext(source)

                if path_ext != ".tp":
                    if path_ext != ".pxd":
                        sources.append(source)
                    continue

                # `source` is a Tempita file
                tempita_sources.append(source)

                # Do not include pxd files that were generated by tempita
                if os.path.splitext(new_source_path)[-1] == ".pxd":
                    continue
                sources.append(new_source_path)

            gen_from_templates(tempita_sources)

            # By convention, our extensions always use the name of the first source
            source_name = os.path.splitext(os.path.basename(sources[0]))[0]
            if submodule:
                name_parts = [submodule, source_name]
            else:
                name_parts = [source_name]
            name = ".".join(name_parts)

            # Make paths start from the root directory
            include_dirs = [
                join(parent_dir, include_dir)
                for include_dir in extension.get("include_dirs", [])
            ]
            if extension.get("include_np", False):
                include_dirs.append(np_include)

            depends = [
                join(parent_dir, depend) for depend in extension.get("depends", [])
            ]

            extra_compile_args = (
                extension.get("extra_compile_args", []) + default_extra_compile_args
            )
            optimization_level = extension.get(
                "optimization_level", default_optimization_level
            )
            if os.name == "posix":
                extra_compile_args.append(f"-{optimization_level}")
            else:
                extra_compile_args.append(f"/{optimization_level}")

            libraries_ext = extension.get("libraries", []) + default_libraries
            new_ext = Extension(
                name=name,
                sources=sources,
                language=extension.get("language", None),
                include_dirs=include_dirs,
                libraries=libraries_ext,
                depends=depends,
                extra_link_args=extension.get("extra_link_args", None),
                extra_compile_args=extra_compile_args,
            )
            new_ext.define_macros.append(DEFINE_MACRO_NUMPY_C_API)
            cython_exts.append(new_ext)
    return cythonize_extensions(cython_exts), metrics, xsimd_archs


def setup_package():
    python_requires = ">=3.8"
    required_python_version = (3, 8)

    metadata = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: C",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Development Status :: 2 - Pre-Alpha",
            # "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: Implementation :: CPython",
        ],
        cmdclass=cmdclass,
        python_requires=python_requires,
        install_requires=tag_to_packages["install"],
        package_data={"": ["*.csv", "*.gz", "*.txt", "*.pxd", "*.rst", "*.jpg"]},
        zip_safe=False,  # the package can run out of an .egg file
        extras_require={
            key: tag_to_packages[key]
            for key in ["examples", "docs", "tests", "benchmark"]
        },
        long_description_content_type="markdown",
    )

    arguments = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if not any(arg in ("egg_info", "dist_info", "clean", "check") for arg in arguments):
        if sys.version_info < required_python_version:
            required_version = "%d.%d" % required_python_version
            raise RuntimeError(
                f"SLSDM requires Python {required_version} or later. The current Python"
                f" version is {platform.python_version()} installed in"
                f" {sys.executable}."
            )

        check_package_status("sklearn", SKLEARN_MIN_VERSION)

        _check_cython_version()
        ext_modules, metrics, xsimd_archs = configure_extension_modules()
        metadata["ext_modules"] = ext_modules
        if metrics is not None and xsimd_archs is not None:
            metadata["libraries"] = _make_library_config(metrics, xsimd_archs)
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
