
#include "xsimd/xsimd.hpp"
#include "generated/manhattan.hpp"
#include "generated/euclidean.hpp"
#include "generated/seuclidean.hpp"
#include "generated/chebyshev.hpp"
#include "generated/minkowski.hpp"
#include "generated/minkowski_w.hpp"

// These must match the functions imported in _dist_metrics.pxd.tp
// ===============================================================

template<typename Type>
auto xsimd_manhattan_dist = xs::dispatch(_manhattan{});

template<typename Type>
auto xsimd_euclidean_rdist = xs::dispatch(_euclidean{});

template<typename Type>
auto xsimd_seuclidean_rdist = xs::dispatch(_seuclidean{});

template<typename Type>
auto xsimd_chebyshev_dist = xs::dispatch(_chebyshev{});

template<typename Type>
auto xsimd_minkowski_rdist = xs::dispatch(_minkowski{});

template<typename Type>
auto xsimd_minkowski_w_rdist = xs::dispatch(_minkowski_w{});
