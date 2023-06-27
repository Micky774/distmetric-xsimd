#include "utils.hpp"

using ARCH_LIST = xs::arch_list<xs::avx512f, xs::sse4_1, xs::ssse3, xs::avx2, xs::avx512dq, xs::sse2, xs::avx, xs::avx512cd, xs::fma3<xs::avx2>, xs::fma3<xs::avx>, xs::fma3<xs::sse4_2>, xs::sse3, xs::avx512bw, xs::sse4_2>;

// These must match the functions imported in _dist_metrics.pxd.tp
// ===============================================================

#include "generated/euclidean.hpp"
template<typename Type>
auto xsimd_euclidean_rdist = xs::dispatch<ARCH_LIST>(_euclidean{});

#include "generated/minkowski_w.hpp"
template<typename Type>
auto xsimd_minkowski_w_rdist = xs::dispatch<ARCH_LIST>(_minkowski_w{});

#include "generated/seuclidean.hpp"
template<typename Type>
auto xsimd_seuclidean_rdist = xs::dispatch<ARCH_LIST>(_seuclidean{});

#include "generated/minkowski.hpp"
template<typename Type>
auto xsimd_minkowski_rdist = xs::dispatch<ARCH_LIST>(_minkowski{});

#include "generated/manhattan.hpp"
template<typename Type>
auto xsimd_manhattan_dist = xs::dispatch<ARCH_LIST>(_manhattan{});

#include "generated/chebyshev.hpp"
template<typename Type>
auto xsimd_chebyshev_dist = xs::dispatch<ARCH_LIST>(_chebyshev{});
