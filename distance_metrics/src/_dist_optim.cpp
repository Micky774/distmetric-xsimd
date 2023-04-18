#define VECTOR_LOOP(body, batch_type) \
    std::size_t inc = batch_type::size; \
    std::size_t loop_iter = inc * 2; \
    std::size_t vec_size = size - size % loop_iter; \
    for(std::size_t idx = 0; idx < vec_size; idx += loop_iter) { \
        body \
    }\

#define REMAINDER_LOOP(body) \
    for(std::size_t idx = vec_size; idx < size; ++idx) { \
        body \
    } \

#define UNROLL_2(UNROLL_BODY) \
    UNROLL_BODY(0) \
    UNROLL_BODY(1) \

#define MAKE_STD_VEC_LOOP(SETUP, BODY, batch_type) \
    UNROLL_2(MANHATTAN_SETUP) \
    VECTOR_LOOP( \
        UNROLL_2(MANHATTAN_BODY), \
        batch_type \
    ) \


#ifdef __SSE3__
    #define HAS_SIMD 1
    #include <cstddef>
    #include "xsimd/xsimd.hpp"
    #include "abs.hpp"

    namespace xs = xsimd;

    /*************************************************************************/
    #define MANHATTAN_BODY(ITER) \
        batch_type simd_x_##ITER = batch_type::load(&a[idx + inc * ITER], xs::unaligned_mode{}); \
        batch_type simd_y_##ITER = batch_type::load(&b[idx + inc * ITER], xs::unaligned_mode{}); \
        sum_##ITER += xsimd_abs(simd_x_##ITER - simd_y_##ITER); \
    
    #define MANHATTAN_SETUP(ITER) \
        batch_type sum_##ITER = batch_type::broadcast(0);

    template <typename Type>
    Type xsimd_manhattan_dist(const Type* a, const Type* b, const std::size_t size){
        using batch_type = xs::batch<Type, xs::best_arch>;
        // instantiate functor
        auto xsimd_abs = _xsimd_abs();
        MAKE_STD_VEC_LOOP(MANHATTAN_SETUP, MANHATTAN_BODY, batch_type)

        sum_0 += sum_1;
        batch_type batch_sum = xs::reduce_add(sum_0);
        float scalar_sum = *(float*)&batch_sum;

        // Remaining part that cannot be vectorize
        REMAINDER_LOOP(scalar_sum += fabs(a[idx] - b[idx]);)
        return scalar_sum;
    }

    /*************************************************************************/
    #define EUCLIDEAN_BODY(ITER) \
        batch_type simd_x_##ITER = batch_type::load(&a[idx + inc * ITER], xs::unaligned_mode{}); \
        batch_type simd_y_##ITER = batch_type::load(&b[idx + inc * ITER], xs::unaligned_mode{}); \
        diff_##ITER = simd_x_##ITER - simd_y_##ITER; \
        sum_##ITER += diff_##ITER * diff_##ITER; \

    #define EUCLIDEAN_SETUP(ITER) \
        batch_type sum_##ITER = batch_type::broadcast(0); \
        batch_type diff_##ITER = batch_type::broadcast(0); \

    template <typename Type>
    Type xsimd_euclidean_rdist(const Type* a, const Type* b, const std::size_t size){
        using batch_type = xs::batch<Type, xs::best_arch>;
        // instantiate functor
        auto xsimd_abs = _xsimd_abs();
        MAKE_STD_VEC_LOOP(EUCLIDEAN_SETUP, EUCLIDEAN_BODY, batch_type)
        sum_0 += sum_1;
        batch_type batch_sum = xs::reduce_add(sum_0);
        float scalar_sum = *(float*)&batch_sum;

        // Remaining part that cannot be vectorize
        REMAINDER_LOOP(scalar_sum += (a[idx] - b[idx]) * (a[idx] - b[idx]);)
        return scalar_sum;
    }
    /*************************************************************************/

#else
    #define HAS_SIMD 0
    #include <cstddef>

    template <typename Type>
    Type xsimd_manhattan_dist(const Type* a, const Type* b, const std::size_t size){return -1}
    template <typename Type>
    Type xsimd_euclidean_rdist(const Type* a, const Type* b, const std::size_t size){return -1}

#endif
