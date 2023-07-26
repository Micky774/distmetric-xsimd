from sklearn.utils._typedefs cimport float32_t, intp_t
import numpy as np

cdef extern from "src/_src/_dist_optim.cpp":
    cdef Type xsimd_euclidean_rdist[Type](Type * x, Type * y, intp_t size) nogil
    cdef Type xsimd_chebyshev_dist[Type](Type * x, Type * y, intp_t size) nogil

cpdef test_euclidean():
    cdef float32_t[:] x = np.array([1, 2, 3], dtype=np.float32)
    cdef float32_t[:] y = np.array([4, 5, 6], dtype=np.float32)
    return xsimd_euclidean_rdist(&x[0], &y[0], 3)

print(test_euclidean())
