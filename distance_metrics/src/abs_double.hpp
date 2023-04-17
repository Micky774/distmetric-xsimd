#include "abs.hpp"
#include "xsimd/xsimd.hpp"

namespace xs = xsimd;

template double_batch _xsimd_abs::operator()<double_batch, xs::sse3, double>(xs::sse3, const double_batch&);