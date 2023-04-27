#ifndef MANHATTAN_HPP
#define MANHATTAN_HPP

#include "utils.hpp"

#define MANHATTAN_BODY(ITER) \
    batch_type simd_x_##ITER = batch_type::load(&a[idx + inc * ITER], xs::unaligned_mode{}); \
    batch_type simd_y_##ITER = batch_type::load(&b[idx + inc * ITER], xs::unaligned_mode{}); \
    sum_##ITER += xs::fabs(simd_x_##ITER - simd_y_##ITER);

#define MANHATTAN_SETUP(ITER) \
    batch_type sum_##ITER = batch_type((Type) 0.);

struct _xsimd_manhattan {
    template <class Arch, typename Type>
    Type operator()(Arch, const Type* a, const Type* b, const std::size_t size);
};

template <class Arch, typename Type>
Type _xsimd_manhattan::operator()(Arch, const Type* a, const Type* b, const std::size_t size){
    using batch_type = xs::batch<Type, Arch>;
    MAKE_STD_VEC_LOOP(MANHATTAN_SETUP, MANHATTAN_BODY, batch_type)

    // Reduction
    sum_0 += sum_1;
    batch_type batch_sum = xs::reduce_add(sum_0);
    double scalar_sum = *(Type*)&batch_sum;

    // Remaining part that cannot be vectorize
    REMAINDER_LOOP(scalar_sum += fabs(a[idx] - b[idx]);)
    return (Type) scalar_sum;
}

// SSE3
extern template float _xsimd_manhattan::operator()<xs::sse3, float>(xs::sse3, const float *, const  float *, const std::size_t);
extern template double _xsimd_manhattan::operator()<xs::sse3, double>(xs::sse3, const double *, const double *, const std::size_t);

// SSE2
extern template float _xsimd_manhattan::operator()<xs::sse2, float>(xs::sse2, const float *, const float *, const std::size_t);
extern template double _xsimd_manhattan::operator()<xs::sse2, double>(xs::sse2, const double *, const double *, const std::size_t);
#else
#endif
