from sklearn.utils._typedefs cimport float64_t, float32_t, int32_t, intp_t

cdef extern from "src/generated/_dist_optim.cpp":
    cdef Type xsimd_manhattan_dist[Type](Type * x, Type * y, intp_t size) nogil
    cdef Type xsimd_euclidean_rdist[Type](Type * x, Type * y, intp_t size) nogil
    cdef Type xsimd_seuclidean_rdist[Type](Type * x, Type * y, intp_t size, const Type * v) nogil
    cdef Type xsimd_chebyshev_dist[Type](Type * x, Type * y, intp_t size) nogil
    cdef Type xsimd_minkowski_rdist[Type](Type * x, Type * y, intp_t size, const double p) nogil
    cdef Type xsimd_minkowski_w_rdist[Type](Type * x, Type * y, intp_t size, const Type * w, const double p) nogil

cdef extern from "xsimd/xsimd.hpp" namespace "xsimd::detail":
    ctypedef struct supported_arch:
        unsigned int sse2
        unsigned int sse3
        unsigned int ssse3
        unsigned int sse4_1
        unsigned int sse4_2
        unsigned int sse4a
        unsigned int fma3_sse
        unsigned int fma4
        unsigned int xop
        unsigned int avx
        unsigned int fma3_avx
        unsigned int avx2
        unsigned int fma3_avx2
        unsigned int avx512f
        unsigned int avx512cd
        unsigned int avx512dq
        unsigned int avx512bw
        unsigned int neon
        unsigned int neon64
        unsigned int sve

cdef extern from "xsimd/xsimd.hpp" namespace "xsimd":
    cdef supported_arch available_architectures() noexcept nogil

cdef extern from *:
    """
    #include "xsimd/xsimd.hpp"

    bool _avx_available(){
        return xsimd::avx::available();
    }

    bool _avx512f_available(){
        return xsimd::avx512f::available();
    }
    """
    cdef bint _avx_available() nogil
    cdef bint _avx512f_available() nogil
