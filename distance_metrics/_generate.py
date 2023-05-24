import glob
from textwrap import dedent, indent
from os.path import join, basename
from pathlib import Path

PY_TAB = "    "
GENERATED_DIR = "distance_metrics/src/generated/"
DEFINITIONS_DIR = "distance_metrics/definitions/"

# XXX: Is there a nicer, more user-friendly approach towards
# FMA{3, 4} instructions?
# TODO: Add documentation regarding instruction priorities/ordering
# All SIMD instructions supported by xsimd
_x86 = [
    "sse2",
    "sse3",
    "ssse3",
    "sse4_1",
    "sse4_2",
    "fma3<xs::sse4_2>",
    "avx",
    "fma3<xs::avx>",
    "avx2",
    "fma3<xs::avx2>",
    "fma4",
    "avx512f",
    "avx512cd",
    "avx512dq",
    "avx512bw",
]
_ARM = [
    "neon",
    "neon64",
]


def _get_arch_id(target_arch):
    target_system = None
    for _ARCH in (_x86, _ARM):
        try:
            target_arch_idx = _ARCH.index(target_arch)
            target_system = _ARCH
        except ValueError:
            pass
    if target_system is None:
        raise ValueError(
            f"Unknown target architecture '{target_arch}' provided; please choose from"
            f" {_x86} for x86 systems, and {_ARM} for ARM systems."
        )
    return target_arch_idx, target_system


def _parse_spec(spec, arch):
    target_arch_idx, target_system = _get_arch_id(arch)
    out = set()

    if "<" in spec:
        fma_version = arch[3] if (len(arch) > 3 and arch[:3] == "fma") else -1
        for a in target_system[:target_arch_idx]:
            # Ensure unsupported/mutually-exclusive FMA features are not enabled
            if "fma" not in a or a[3] == fma_version:
                out |= {a}
    if "<=" in spec or not spec:
        out |= {target_system[target_arch_idx]}
    if "!" in spec:
        out -= {target_system[target_arch_idx]}
    return out


def _make_architectures(target_archs):
    SPECIFIERS = ["<=", "<", "!"]
    out = set()
    for config in target_archs.split(","):
        config = config.strip()
        spec = ""
        arch = config
        for mark in SPECIFIERS:
            # Guard against incorrectly parsing fma3<...>
            if mark in config[:2]:
                spec = mark
                arch = arch[len(spec) :]
                break

        out |= _parse_spec(spec, arch)
    return list(out)


def _pprint_config(config):
    for key in config:
        print(f"For function {key}:\n")
        spec = config[key]
        for section in spec:
            print(f"Showing section: {section}:\n")
            print(spec[section])
        print(f"{'':=^80}")


def get_config():
    definitions = glob.glob(join(DEFINITIONS_DIR, r"*.def"))
    config = {}
    for def_file_name in definitions:
        mode = None
        with open(def_file_name) as file:
            specification = {"SETUP": None, "BODY": None, "REMAINDER": None}
            section = ""
            for line in file:
                line = line.rstrip()
                if line in ("SETUP", "BODY", "REMAINDER"):
                    if mode is not None:
                        specification[mode] = section
                    section = ""
                    mode = line
                else:
                    section += line + "\n"
            specification[mode] = section
        function_name = basename(def_file_name)[:-4]
        config[function_name] = specification
    return config


def gen_from_config(config, target_arch):
    # TODO: Parse definition files directly in python rather than relying on C
    # macros
    ARCHITECTURES = _make_architectures(target_arch)
    print(f"Generating the following SIMD targets: {ARCHITECTURES}...\n")

    file_template = dedent("""\
        #ifndef {1}_HPP
        #define {1}_HPP
        #include "utils.hpp"

        struct _{0}{{
        template <class Arch, typename Type>
        Type operator()(Arch, const Type* a, const Type* b, const std::size_t size);
        }};

        #define {1}_SETUP(ITER) \\
        {2}
        #define {1}_BODY(ITER) \\
        {3}
        template <class Arch, typename Type>
        Type _{0}::operator()(Arch, const Type* a, const Type* b, const std::size_t size){{
            using batch_type = xs::batch<Type, Arch>;
            MAKE_STD_VEC_LOOP({1}_SETUP, {1}_BODY, batch_type)

        {4}
        }}
        """)  # noqa

    target_specific_templates = {}
    for arch in ARCHITECTURES:
        target_specific_templates[arch] = """#include "{0}.hpp"\n"""
        file_template += "\n"
        file_template += f"// {arch.upper()}\n"
        for type in ("float", "double"):
            signature = f"template {type} _{{0}}::operator()<xs::{arch}, {type}>(xs::{arch}, const {type} *, const  {type} *, const std::size_t);\n"  # noqa
            target_specific_templates[arch] += signature
            file_template += "extern " + signature

    file_template += "#else\n#endif /* {1}_HPP */"
    for metric, spec in config.items():
        file_content = file_template.format(
            metric,
            metric.upper(),
            indent(spec["SETUP"], PY_TAB),
            indent(spec["BODY"], PY_TAB),
            indent(spec["REMAINDER"], PY_TAB),
        )
        file_path = join(GENERATED_DIR, f"{metric}.hpp")
        with open(file_path, "w") as file:
            file.write(file_content)

        for arch in ARCHITECTURES:
            file_path = join(GENERATED_DIR, f"{metric}_{arch}.cpp")
            with open(file_path, "w") as file:
                file.write(target_specific_templates[arch].format(metric))


def generate_code(target_arch):
    # TODO: First check to see whether any source files have been modified and
    # actually require to be regenerated, or an environment flag specifying
    # such has been set.
    Path(GENERATED_DIR).mkdir(parents=True, exist_ok=True)
    gen_from_config(get_config(), target_arch)
