#include "xsimd/xsimd.hpp"

namespace xs = xsimd;

struct xsimd_abs {
    template <class Batch, class Arch, typename Type>
    Batch operator()(Arch, const Batch& x);
};

template <class Batch, class Arch, typename Type>
Batch xsimd_abs::operator()(Arch, const Batch& x){
    using batch_type = xs::batch<Type, Arch>;
    batch_type mask = batch_type::broadcast(-0.f);
    return xs::bitwise_andnot(mask, x);
}

using float_batch = xs::batch<float, xs::sse3>;
using double_batch = xs::batch<double, xs::sse3>;
extern template xs::batch<float, xs::sse3> xsimd_abs::operator()<float_batch, xs::sse3, float>(xs::sse3, const float_batch&);
extern template xs::batch<double, xs::sse3> xsimd_abs::operator()<double_batch, xs::sse2, float>(xs::sse2, const double_batch&);
