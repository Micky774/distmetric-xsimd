#ifdef __SSE3__
    #define HAS_SIMD 1
    #include <cstddef>
    #include <vector>
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

    template <typename Type>
    Type xsimd_manhattan(const Type* a, const Type* b, const std::size_t size){
        using batch_type = xs::batch<Type, xs::best_arch>;
        std::size_t inc = batch_type::size;
        // Multiply inc by two for unrolling
        std::size_t loop_iter = inc * 2;
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % loop_iter;

        auto xsimd_abs = _xsimd_abs();

        batch_type sum_1 = batch_type::broadcast(0);
        batch_type sum_2 = batch_type::broadcast(0);
        for(std::size_t idx = 0; idx < vec_size; idx += loop_iter)
        {
            batch_type simd_x_1 = batch_type::load(&a[idx], xs::unaligned_mode{});
            batch_type simd_y_1 = batch_type::load(&b[idx], xs::unaligned_mode{});
            sum_1 += xsimd_abs(simd_x_1 - simd_y_1);

            batch_type simd_x_2 = batch_type::load(&a[idx + inc], xs::unaligned_mode{});
            batch_type simd_y_2 = batch_type::load(&b[idx + inc], xs::unaligned_mode{});
            sum_2 += xsimd_abs(simd_x_2 - simd_y_2);
        }
        // xs::store(&res[idx], rvec);
        sum_1 += sum_2;
        batch_type batch_sum = xs::reduce_add(sum_1);
        float scalar_sum = *(float*)&batch_sum;
        // Remaining part that cannot be vectorize
        for(std::size_t idx = vec_size; idx < size; ++idx)
        {
            scalar_sum += fabs(a[idx] - b[idx]);
        }
        return scalar_sum;
    }

#else
    #define HAS_SIMD 0
    #include <cstddef>

    template <typename Type>
    Type xsimd_manhattan(const Type* a, const Type* b, const std::size_t size){return -1}
#endif
