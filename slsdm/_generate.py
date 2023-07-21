import glob
from textwrap import dedent, indent
from os.path import join, basename
from pathlib import Path
import io

PY_TAB = "    "
GENERATED_DIR = "slsdm/src/generated/"
DEFINITIONS_DIR = "slsdm/definitions/"
VECTOR_UNROLL_FACTOR = 4

# TODO: Add documentation regarding instruction priorities/ordering
# All SIMD instructions supported by xsimd
_x86 = [
    "avx512bw",
    "avx512dq",
    "avx512cd",
    "avx512f",
    "fma4",
    "avx2",
    "avx",
    "sse4_2",
    "sse4_1",
    "ssse3",
    "sse3",
    "sse2",
]


def _parse_arch_flag(arch):
    if "fma3" in arch:
        return "fma"
    return arch.replace("_", ".")


def _get_arch_id(target_arch):
    try:
        target_arch_idx = _x86.index(target_arch)
    except ValueError:
        raise ValueError(
            f"Unknown target architecture '{target_arch}' provided; please choose from"
            f" {_x86} for x86 systems. Note we do not currently support ARM systems."
        )
    return target_arch_idx


def _parse_spec(spec, arch, cur_archs):
    target_arch_idx = _get_arch_id(arch)
    if "<" in spec:
        fma_version = arch[3] if (len(arch) > 3 and arch[:3] == "fma") else -1
        for a in _x86[target_arch_idx:]:
            # Ensure unsupported/mutually-exclusive FMA features are not enabled
            if "fma" not in a or a[3] == fma_version:
                cur_archs[a] = None
    if spec == "<=" or not spec:
        cur_archs[_x86[target_arch_idx]] = None
    if spec == "~":
        cur_archs.pop(_x86[target_arch_idx], None)
    return cur_archs


def _make_architectures(target_archs):
    SPECIFIERS = ["<=", "<", "~"]
    cur_archs = {}
    has_fma3 = False
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
        if arch == "fma3":
            has_fma3 = True
        else:
            _parse_spec(spec, arch, cur_archs)
    cur_archs = list(cur_archs)

    # Second pass to enable any compatible fma3 sets
    if has_fma3:
        for fma_compatible_arch in ("sse4_2", "avx", "avx2"):
            if fma_compatible_arch in cur_archs:
                cur_archs.append(f"fma3<xs::{fma_compatible_arch}>")

    return cur_archs


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
        "DIST_TYPE",
        "N_UNROLL",
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
for(std::size_t idx = vec_remainder_size; idx < size; ++idx) {{
{_tab_indent(body)}
}}"""


def _UNROLL(UNROLL_BODY, n_unroll):
    out = "// Begin unrolled\n"
    for i in range(n_unroll):
        out += f"// Loop #{i}\n{UNROLL_BODY(i)}\n"
    out += "// End unrolled\n"
    return out


def _MAKE_STD_VEC_LOOP(SETUP, BODY, n_unroll):
    return f"""\
// Begin SETUP
{_UNROLL(SETUP, n_unroll)}
// End SETUP

// Begin VECTOR LOOP
std::size_t inc = batch_type::size;
std::size_t loop_iter = inc * {n_unroll};
std::size_t vec_size = size - size % loop_iter;
std::size_t vec_remainder_size = size - size % inc;
for(std::size_t idx = 0; idx < vec_size; idx += loop_iter) {{
{indent(_UNROLL(BODY, n_unroll), PY_TAB)}
}}
for(std::size_t idx = vec_size; idx < vec_remainder_size; idx += inc) {{
{indent(BODY(0), PY_TAB)}
}}
// End VECTOR LOOP"""


def gen_from_config(config, target_arch):
    # TODO: Parse definition files directly in python rather than relying on C
    # macros
    ARCHITECTURES = _make_architectures(target_arch)
    print(f"Generating the following SIMD targets: {ARCHITECTURES}\n")

    FILE_TEMPLATE = dedent("""\
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

    feature_flags = []
    xsimd_archs = ""
    for arch in ARCHITECTURES:
        xsimd_archs += f"xs::{arch}, "
        flag = f"-m{_parse_arch_flag(arch)}"
        feature_flags.append(flag)
    xsimd_archs = xsimd_archs[:-2]

    optim_file = dedent(f"""\
        #include "utils.hpp"

        using ARCH_LIST = xs::arch_list<{xsimd_archs}>;

        // These must match the functions imported in _dist_metrics.pxd.tp
        // ===============================================================

        """)

    def _write_arch_specialization(metric, arch, signature_template, additional_args):
        file_path = join(GENERATED_DIR, f"{metric}_{arch}.cpp")
        arch_specialized_template = """#include "{0}.hpp"\n"""
        for _type in ("float", "double"):
            signature = (
                signature_template.format(metric, additional_args, arch)
                .replace("Type", _type)
                .replace("{", ";")
            )
            arch_specialized_template += signature
        with open(file_path, "w") as file:
            file.write(arch_specialized_template.format(metric))
        return "extern " + signature

    def _specialize_file_content(metric, spec):
        setup_func = lambda n: _make_parseable(spec["SETUP_UNROLL"]).format(n)
        body_func = lambda n: _make_parseable(spec["BODY"]).format(n)
        additional_args = ", " + spec["ARGS"] if spec["ARGS"] else ""
        file_content = FILE_TEMPLATE.format(
            metric,
            additional_args,
            metric.upper(),
            _tab_indent(spec["SETUP"] if spec["SETUP"] else ""),
            _tab_indent(
                _MAKE_STD_VEC_LOOP(setup_func, body_func, int(spec["N_UNROLL"]))
            ),
            _tab_indent(spec["REDUCTION"]),
            _tab_indent(_REMAINDER_LOOP(spec["REMAINDER"])),
            indent(spec["OUT"], PY_TAB),
        )
        return file_content, additional_args

    signature_template = (
        "template " + io.StringIO(FILE_TEMPLATE).readlines()[10]
    ).replace("operator()(Arch", "operator()<xs::{2}, Type>(xs::{2}")

    for metric, spec in config.items():
        optim_file += dedent(f"""\
            #include "{metric}.hpp"
            template<typename Type>
            auto xsimd_{metric}_{spec["DIST_TYPE"]} = xs::dispatch<ARCH_LIST>(_{metric}{{}});

            """)  # noqa
        file_content, additional_args = _specialize_file_content(metric, spec)

        for arch in ARCHITECTURES:
            # Keep track of extern statements as we write specializations
            file_content += _write_arch_specialization(
                metric, arch, signature_template, additional_args
            )

        file_content += f"#else\n#endif /* {metric.upper()}_HPP */"
        file_path = join(GENERATED_DIR, f"{metric}.hpp")
        with open(file_path, "w") as file:
            file.write(file_content)

    file_path = join(GENERATED_DIR, "_dist_optim.cpp")
    with open(file_path, "w") as file:
        file.write(optim_file)

    return feature_flags


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
    return gen_from_config(get_config(), target_arch)
