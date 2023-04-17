#ifndef ABS_HPP
#define ABS_HPP
#include <cstddef>
#include "xsimd/xsimd.hpp"

namespace xs = xsimd;

struct _xsimd_abs {
    template <class Batch>
    Batch operator()(const Batch& x){
        using batch_type = Batch;
        batch_type mask = batch_type::broadcast(-0.f);
        return xs::bitwise_andnot(mask, x);
    }
};
#endif