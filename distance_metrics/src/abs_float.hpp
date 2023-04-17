#include "abs.hpp"
#include "xsimd/xsimd.hpp"

namespace xs = xsimd;

template float_batch xsimd_abs::operator()<float_batch, xs::sse3, float>(xs::sse3, const float_batch&);