
#include "xsimd/xsimd.hpp"
#include "generated/manhattan.hpp"
#include "generated/euclidean.hpp"

template<typename Type>
auto xsimd_manhattan_dist = xs::dispatch(_manhattan{});

template<typename Type>
auto xsimd_euclidean_rdist = xs::dispatch(_euclidean{});
