#ifndef CHEBYSHEV_HPP
#define CHEBYSHEV_HPP
#include "utils.hpp"

struct _chebyshev{
template <class Arch, typename Type>
Type operator()(Arch, const Type* a, const Type* b, const std::size_t size);
};

template <class Arch, typename Type>
Type _chebyshev::operator()(Arch, const Type* a, const Type* b, const std::size_t size){
    using batch_type = xs::batch<Type, Arch>;

    // Begin SETUP
    // Begin unrolled
    // Loop #0
    batch_type simd_x_0;
    batch_type simd_y_0;
    batch_type segment_0 = batch_type::broadcast(0.);
    // Loop #1
    batch_type simd_x_1;
    batch_type simd_y_1;
    batch_type segment_1 = batch_type::broadcast(0.);
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
        segment_0 = xs::max(segment_0, xs::abs(simd_x_0 - simd_y_0));
        // Loop #1
        simd_x_1 = batch_type::load_unaligned(&a[idx + inc * 1]);
        simd_y_1 = batch_type::load_unaligned(&b[idx + inc * 1]);
        segment_1 = xs::max(segment_1, xs::abs(simd_x_1 - simd_y_1));
        // End unrolled

    }
    for(std::size_t idx = vec_size; idx < vec_remainder_size; idx += inc) {
        simd_x_0 = batch_type::load_unaligned(&a[idx + inc * 0]);
        simd_y_0 = batch_type::load_unaligned(&b[idx + inc * 0]);
        segment_0 = xs::max(segment_0, xs::abs(simd_x_0 - simd_y_0));
    }
    // End VECTOR LOOP
    // Reduction
    segment_0 = xs::reduce_max(xs::max(segment_0, segment_1));
    double scalar_max = *(Type*)&segment_0;

    // Remaining part that cannot be vectorize
    for(std::size_t idx = vec_remainder_size; idx < size; ++idx) {
        scalar_max = fmax(scalar_max, fabs(a[idx] - b[idx]));
    }
    return (Type) scalar_max;
}
extern template float _chebyshev::operator()<xs::avx512f, float>(xs::avx512f, const float* a, const float* b, const std::size_t size);
extern template float _chebyshev::operator()<xs::avx, float>(xs::avx, const float* a, const float* b, const std::size_t size);
extern template float _chebyshev::operator()<xs::sse3, float>(xs::sse3, const float* a, const float* b, const std::size_t size);
extern template float _chebyshev::operator()<xs::sse2, float>(xs::sse2, const float* a, const float* b, const std::size_t size);


extern template double _chebyshev::operator()<xs::avx512f, double>(xs::avx512f, const double* a, const double* b, const std::size_t size);
extern template double _chebyshev::operator()<xs::avx, double>(xs::avx, const double* a, const double* b, const std::size_t size);
extern template double _chebyshev::operator()<xs::sse3, double>(xs::sse3, const double* a, const double* b, const std::size_t size);
extern template double _chebyshev::operator()<xs::sse2, double>(xs::sse2, const double* a, const double* b, const std::size_t size);
#else
#endif /* CHEBYSHEV_HPP */
