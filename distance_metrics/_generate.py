import glob
from textwrap import dedent, indent
from os.path import join, basename
from pathlib import Path
import io

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
    SECTIONS = (
        "ARGS",
        "SETUP",
        "SETUP_UNROLL",
        "BODY",
        "REDUCTION",
        "REMAINDER",
        "OUT",
    )
    definitions = glob.glob(join(DEFINITIONS_DIR, r"*.def"))
    config = {}
    for def_file_name in definitions:
        mode = None
        with open(def_file_name) as file:
            specification = {section: None for section in SECTIONS}
            section = ""
            for line in file:
                line = line.rstrip()
                if line in SECTIONS:
                    if mode is not None:
                        specification[mode] = section.strip()
                    section = ""
                    mode = line
                else:
                    section += line + "\n"
            specification[mode] = section.strip()
        function_name = basename(def_file_name)[:-4]
        config[function_name] = specification
    return config


# TODO: Use dedent to make this a bit more readable
def _REMAINDER_LOOP(body):
    return f"""\
for(std::size_t idx = vec_size; idx < size; ++idx) {{
{_tab_indent(body)}
}}"""


def _UNROLL_2(UNROLL_BODY):
    return f"""\
// Begin unrolled
{UNROLL_BODY(0)}

{UNROLL_BODY(1)}
// End unrolled"""


def _MAKE_STD_VEC_LOOP(SETUP, BODY):
    return f"""\
// Begin SETUP
{_UNROLL_2(SETUP)}
// End SETUP

// Begin VECTOR LOOP
std::size_t inc = batch_type::size;
std::size_t loop_iter = inc * 2;
std::size_t vec_size = size - size % loop_iter;
for(std::size_t idx = 0; idx < vec_size; idx += loop_iter) {{
{indent(_UNROLL_2(BODY), PY_TAB)}
}}
// End VECTOR LOOP"""


def gen_from_config(config, target_arch):
    # TODO: Parse definition files directly in python rather than relying on C
    # macros
    ARCHITECTURES = _make_architectures(target_arch)
    print(f"Generating the following SIMD targets: {ARCHITECTURES}\n")

    file_template = dedent("""\
        #ifndef {2}_HPP
        #define {2}_HPP
        #include "utils.hpp"

        struct _{0}{{
        template <class Arch, typename Type>
        Type operator()(Arch, const Type* a, const Type* b, const std::size_t size{1});
        }};

        template <class Arch, typename Type>
        Type _{0}::operator()(Arch, const Type* a, const Type* b, const std::size_t size{1}){{
            using batch_type = xs::batch<Type, Arch>;
        {3}
        {4}
        {5}

            // Remaining part that cannot be vectorize
        {6}
        {7}
        }}
        """)  # noqa
    signature_template = "template " + io.StringIO(file_template).readlines()[
        10
    ].replace("operator()(Arch", "operator()<xs::{2}, Type>(xs::{2}")
    for metric, spec in config.items():
        setup_func = lambda n: _make_parseable(spec["SETUP_UNROLL"]).format(n)
        body_func = lambda n: _make_parseable(spec["BODY"]).format(n)
        additional_args = ", " + spec["ARGS"] if spec["ARGS"] else ""
        file_content = file_template.format(
            metric,
            additional_args,
            metric.upper(),
            _tab_indent(spec["SETUP"] if spec["SETUP"] else ""),
            _tab_indent(_MAKE_STD_VEC_LOOP(setup_func, body_func)),
            _tab_indent(spec["REDUCTION"]),
            _tab_indent(_REMAINDER_LOOP(spec["REMAINDER"])),
            indent(spec["OUT"], PY_TAB),
        )

        for arch in ARCHITECTURES:
            file_path = join(GENERATED_DIR, f"{metric}_{arch}.cpp")
            target_specific_template = """#include "{0}.hpp"\n"""
            for _type in ("float", "double"):
                signature = (
                    signature_template.format(metric, additional_args, arch)
                    .replace("Type", _type)
                    .replace("{", ";")
                )
                target_specific_template += signature
                file_content += "extern " + signature
            with open(file_path, "w") as file:
                file.write(target_specific_template.format(metric))

        file_path = join(GENERATED_DIR, f"{metric}.hpp")
        file_content += "#else\n#endif /* {metric.upper()}_HPP */"
        with open(file_path, "w") as file:
            file.write(file_content)


def _tab_indent(str):
    return indent(str, PY_TAB)


def _make_parseable(raw):
    return (
        raw.strip()
        .replace("{", "{{")
        .replace("}", "}}")
        .replace("##ITER", "{0}")
        .replace("ITER", "{0}")
    )


def generate_code(target_arch):
    # TODO: First check to see whether any source files have been modified and
    # actually require to be regenerated, or an environment flag specifying
    # such has been set.
    Path(GENERATED_DIR).mkdir(parents=True, exist_ok=True)
    gen_from_config(get_config(), target_arch)
