#include "manhattan.hpp"

template float _xsimd_manhattan::operator()<xs::sse3, float>(xs::sse3, const float *, const  float *, const std::size_t);
template double _xsimd_manhattan::operator()<xs::sse3, double>(xs::sse3, const double *, const double *, const std::size_t);
