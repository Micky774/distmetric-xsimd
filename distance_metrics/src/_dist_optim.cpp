
#include <iostream>
#include <cstddef>
#include "xsimd/xsimd.hpp"
#include "manhattan.hpp"
#include "utils.hpp"

namespace xs = xsimd;
using chosen_arch = xs::sse3;


/*************************************************************************/
template<typename Type>
auto xsimd_manhattan_dist = xs::dispatch(_xsimd_manhattan{});

/*************************************************************************/
#define EUCLIDEAN_BODY(ITER) \
    batch_type simd_x_##ITER = batch_type::load(&a[idx + inc * ITER], xs::unaligned_mode{}); \
    batch_type simd_y_##ITER = batch_type::load(&b[idx + inc * ITER], xs::unaligned_mode{}); \
    diff_##ITER = simd_x_##ITER - simd_y_##ITER; \
    sum_##ITER += diff_##ITER * diff_##ITER;

#define EUCLIDEAN_SETUP(ITER) \
    batch_type sum_##ITER = batch_type::broadcast(0); \
    batch_type diff_##ITER = batch_type::broadcast(0);

template <typename Type>
Type xsimd_euclidean_rdist(const Type* a, const Type* b, const std::size_t size){
    using batch_type = xs::batch<Type, chosen_arch>;
    MAKE_STD_VEC_LOOP(EUCLIDEAN_SETUP, EUCLIDEAN_BODY, batch_type)

    // Reduction
    sum_0 += sum_1;
    batch_type batch_sum = xs::reduce_add(sum_0);
    Type scalar_sum = *(Type*)&batch_sum;

    // Remaining part that cannot be vectorize
    REMAINDER_LOOP(scalar_sum += (a[idx] - b[idx]) * (a[idx] - b[idx]);)
    return scalar_sum;
}
