#ifndef UTILS_HPP
#define UTILS_HPP

#include "xsimd/xsimd.hpp"

namespace xs = xsimd;

#define VECTOR_LOOP(body, batch_type, n_unroll) \
    std::size_t inc = batch_type::size; \
    std::size_t loop_iter = inc * n_unroll; \
    std::size_t vec_size = size - size % loop_iter; \
    for(std::size_t idx = 0; idx < vec_size; idx += loop_iter) { \
        body \
    }

#define REMAINDER_LOOP(body) \
    for(std::size_t idx = vec_size; idx < size; ++idx) { \
        body \
    }

#define UNROLL_2(UNROLL_BODY) \
    UNROLL_BODY(0) \
    UNROLL_BODY(1)

#define MAKE_STD_VEC_LOOP(SETUP, BODY, batch_type) \
    UNROLL_2(SETUP) \
    VECTOR_LOOP( \
        UNROLL_2(BODY), \
        batch_type, \
        2 \
    )

// AVX2
using float_batch_avx2 = xs::batch<float, xs::avx2>;
using double_batch_avx2 = xs::batch<double, xs::avx2>;

// SSE3
using float_batch_sse3 = xs::batch<float, xs::sse3>;
using double_batch_sse3 = xs::batch<double, xs::sse3>;

// SSE2
using float_batch_sse2 = xs::batch<float, xs::sse2>;
using double_batch_sse2 = xs::batch<double, xs::sse2>;

#else
#endif
