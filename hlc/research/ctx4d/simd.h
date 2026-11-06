// simd.h - native AVX2 when the compiler targets it, otherwise the SIMDe amalgam bundled in the plt2 root
// (../../../avx2.h, SIMDe 0.8.2): SSE-only x86 such as Ivy Bridge, ARM NEON, ... -DFORCE_SIMDE uses SIMDe always.
#pragma once
#include <cstdlib>
#if defined(__AVX2__) && !defined(FORCE_SIMDE)
  #include <immintrin.h>
  #define CTX4D_SIMD "native AVX2"
#else
  #define SIMDE_ENABLE_NATIVE_ALIASES
  #include "../../../avx2.h"
  #define CTX4D_SIMD "SIMDe AVX2 emulation"
#endif
#if defined(_MSC_VER)
  #include <malloc.h>
  static inline void* simd_alloc(size_t a, size_t n) { return _aligned_malloc(n, a); }
  static inline void simd_free(void* p) { _aligned_free(p); }
#else
  static inline void* simd_alloc(size_t a, size_t n) { return aligned_alloc(a, (n + a - 1) / a * a); }
  static inline void simd_free(void* p) { free(p); }
#endif
