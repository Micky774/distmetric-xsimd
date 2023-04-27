#include "manhattan.hpp"

template float _xsimd_manhattan::operator()<xs::sse2, float>(xs::sse2, const float *, const float *, const std::size_t);
template double _xsimd_manhattan::operator()<xs::sse2, double>(xs::sse2, const double *, const double *, const std::size_t);
