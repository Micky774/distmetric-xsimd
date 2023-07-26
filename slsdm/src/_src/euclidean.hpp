#ifndef EUCLIDEAN_HPP
#define EUCLIDEAN_HPP
#include "utils.hpp"

struct _euclidean{
template <class Arch, typename Type>
Type operator()(Arch, const Type* a, const Type* b, const std::size_t size);
};

template <class Arch, typename Type>
Type _euclidean::operator()(Arch, const Type* a, const Type* b, const std::size_t size){
    using batch_type = xs::batch<Type, Arch>;

    // Begin SETUP
    // Begin unrolled
    // Loop #0
    batch_type simd_x_0;
    batch_type simd_y_0;
    batch_type sum_0 = batch_type::broadcast(0.);
    batch_type diff_0 = batch_type::broadcast(0.);
    // Loop #1
    batch_type simd_x_1;
    batch_type simd_y_1;
    batch_type sum_1 = batch_type::broadcast(0.);
    batch_type diff_1 = batch_type::broadcast(0.);
    // End unrolled

    // End SETUP

    // Begin VECTOR LOOP
    std::size_t inc = batch_type::size;
    std::size_t loop_iter = inc * 2;
    std::size_t vec_size = size - size % loop_iter;
    std::size_t vec_remainder_size = size - size % inc;
    for(std::size_t idx = 0; idx < vec_size; idx += loop_iter) {
        // Begin unrolled
        // Loop #0
        simd_x_0 = batch_type::load_unaligned(&a[idx + inc * 0]);
        simd_y_0 = batch_type::load_unaligned(&b[idx + inc * 0]);
        diff_0 = simd_x_0 - simd_y_0;
        sum_0 += diff_0 * diff_0;
        // Loop #1
        simd_x_1 = batch_type::load_unaligned(&a[idx + inc * 1]);
        simd_y_1 = batch_type::load_unaligned(&b[idx + inc * 1]);
        diff_1 = simd_x_1 - simd_y_1;
        sum_1 += diff_1 * diff_1;
        // End unrolled

    }
    for(std::size_t idx = vec_size; idx < vec_remainder_size; idx += inc) {
        simd_x_0 = batch_type::load_unaligned(&a[idx + inc * 0]);
        simd_y_0 = batch_type::load_unaligned(&b[idx + inc * 0]);
        diff_0 = simd_x_0 - simd_y_0;
        sum_0 += diff_0 * diff_0;
    }
    // End VECTOR LOOP
    // Reduction
    sum_0 += sum_1;
    batch_type batch_sum = xs::reduce_add(sum_0);
    double scalar_sum = *(Type*)&batch_sum;

    // Remaining part that cannot be vectorize
    for(std::size_t idx = vec_remainder_size; idx < size; ++idx) {
        scalar_sum += (a[idx] - b[idx]) * (a[idx] - b[idx]);
    }
    return (Type) scalar_sum;
}
extern template float _euclidean::operator()<xs::avx512f, float>(xs::avx512f, const float* a, const float* b, const std::size_t size);
extern template float _euclidean::operator()<xs::avx, float>(xs::avx, const float* a, const float* b, const std::size_t size);
extern template float _euclidean::operator()<xs::sse3, float>(xs::sse3, const float* a, const float* b, const std::size_t size);
extern template float _euclidean::operator()<xs::sse2, float>(xs::sse2, const float* a, const float* b, const std::size_t size);


extern template double _euclidean::operator()<xs::avx512f, double>(xs::avx512f, const double* a, const double* b, const std::size_t size);
extern template double _euclidean::operator()<xs::avx, double>(xs::avx, const double* a, const double* b, const std::size_t size);
extern template double _euclidean::operator()<xs::sse3, double>(xs::sse3, const double* a, const double* b, const std::size_t size);
extern template double _euclidean::operator()<xs::sse2, double>(xs::sse2, const double* a, const double* b, const std::size_t size);
#else
#endif /* EUCLIDEAN_HPP */
