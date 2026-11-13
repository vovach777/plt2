// as_rt.hpp - wasm32-like runtime for as2cpp.py output: one linear memory of 64 KiB pages (reserved 4 GiB, so pointers
// never move), little-endian unaligned load / store, clz / ctz / popcnt with wasm results for 0.
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <sys/mman.h>
static uint8_t* MEM = [] { void* p = mmap(nullptr, size_t(1) << 32, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) abort(); return static_cast<uint8_t*>(p); }();
static int32_t g_pages = 256;
static inline int32_t mem_size() { return g_pages; }
static inline int32_t mem_grow(int32_t n) { const int32_t o = g_pages; if (int64_t(g_pages) + n > 65536) return -1; g_pages += n; return o; }
static inline void chk(uint32_t p, uint32_t n) { if (uint64_t(p) + n > uint64_t(g_pages) << 16) { fprintf(stderr, "out of bounds memory access at %u\n", p); abort(); } }
template <class T> static inline T ld(uint32_t p) { chk(p, sizeof(T)); T v; memcpy(&v, MEM + p, sizeof(T)); return v; }
template <class T, class V> static inline void st(uint32_t p, V v) { chk(p, sizeof(T)); const T t = T(v); memcpy(MEM + p, &t, sizeof(T)); }
static inline void mem_copy(uint32_t d, uint32_t s, uint32_t n) { chk(d, n); chk(s, n); memmove(MEM + d, MEM + s, n); }
static inline void mem_fill(uint32_t d, uint32_t v, uint32_t n) { chk(d, n); memset(MEM + d, int(v), n); }
static inline uint32_t clz_32(uint32_t x) { return x ? __builtin_clz(x) : 32; }
static inline uint64_t clz_64(uint64_t x) { return x ? __builtin_clzll(x) : 64; }
static inline uint32_t ctz_32(uint32_t x) { return x ? __builtin_ctz(x) : 32; }
static inline uint64_t ctz_64(uint64_t x) { return x ? __builtin_ctzll(x) : 64; }
static inline uint32_t popcnt_32(uint32_t x) { return __builtin_popcount(x); }
static inline uint64_t popcnt_64(uint64_t x) { return __builtin_popcountll(x); }
#include <cstdio>
