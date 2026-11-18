// Совместимость заголовков FlashInfer с MSVC: подставляется nvcc --pre-include.
#pragma once

#if defined(_MSC_VER)
// На Linux ushort приходит из sys/types.h, в MSVC/CUDA такого типа нет.
#ifndef ushort
typedef unsigned short ushort;
#endif
// GCC-атрибуты в заголовках FlashInfer (always_inline и подобные) MSVC не знает.
#ifndef __attribute__
#define __attribute__(x)
#endif
// MSVC зовёт квалификатор __restrict, а не __restrict__.
#ifndef __restrict__
#define __restrict__ __restrict
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif
#endif

#include <type_traits>
namespace fi_win {
template <class A, class B>
__host__ __device__ constexpr typename std::common_type<A, B>::type fi_max(A a, B b) {
    using T = typename std::common_type<A, B>::type;
    return static_cast<T>(a) > static_cast<T>(b) ? static_cast<T>(a) : static_cast<T>(b);
}
template <class A, class B>
__host__ __device__ constexpr typename std::common_type<A, B>::type fi_min(A a, B b) {
    using T = typename std::common_type<A, B>::type;
    return static_cast<T>(a) < static_cast<T>(b) ? static_cast<T>(a) : static_cast<T>(b);
}
}
#define fi_max fi_win::fi_max
#define fi_min fi_win::fi_min
