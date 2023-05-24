
#include "xsimd/xsimd.hpp"
#include "generated/manhattan.hpp"
#include "generated/euclidean.hpp"
#include "generated/chebyshev.hpp"

// These must match the functions imported in _dist_metrics.pxd.tp
// ===============================================================

template<typename Type>
auto xsimd_manhattan_dist = xs::dispatch(_manhattan{});

template<typename Type>
auto xsimd_euclidean_rdist = xs::dispatch(_euclidean{});

template<typename Type>
auto xsimd_chebyshev_dist = xs::dispatch(_chebyshev{});
