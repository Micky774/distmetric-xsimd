from sklearn.utils._typedefs cimport float64_t, float32_t, int32_t, intp_t

cdef extern from "src/generated/_dist_optim.cpp":
    cdef Type xsimd_manhattan_dist[Type](Type * x, Type * y, intp_t size) nogil
    cdef Type xsimd_euclidean_rdist[Type](Type * x, Type * y, intp_t size) nogil
    cdef Type xsimd_seuclidean_rdist[Type](Type * x, Type * y, intp_t size, const Type * v) nogil
    cdef Type xsimd_chebyshev_dist[Type](Type * x, Type * y, intp_t size) nogil
    cdef Type xsimd_minkowski_rdist[Type](Type * x, Type * y, intp_t size, const double p) nogil
    cdef Type xsimd_minkowski_w_rdist[Type](Type * x, Type * y, intp_t size, const Type * w, const double p) nogil

cdef extern from *:
    """
    #include "xsimd/xsimd.hpp"
    #include <iostream>
    using namespace std;

    void _get_best_arch()
    {
        // Print standard output
        // on the screen
        cout << "The best architecture supported by this build is: "\
        << xsimd::default_arch::name();
        return;
    }
    """
    cdef void _get_best_arch() nogil
