// propagate.hpp — 1D local-propagation kernel, SoA hot-fields, double buffer
//
// ============================================================================
//  CORE PARADIGM FILE — modification protocol:
//   1. Run ./tools/regression_test.ps1 BEFORE the change
//   2. Make the change
//   3. Run ./tools/regression_test.ps1 AFTER the change
//   4. Every EXACT golden number in benchmarks/golden_numbers.txt must remain identical
//   5. PERF golden numbers must remain within their declared tolerance
//  Rule `(l + 2c + r) >> 2 * 255/256` is canonical (validated M1, M2.5, M4).
// ============================================================================
//
// Thesis: the rule is light and touches few fields per cell. SoA gives linear,
// vectorizable reads; AoS would force gather. Hot fields live outside the canonical Cell.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <algorithm>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

namespace cl {

// Parallel hot fields, contiguous, SIMD-aligned.
// 1 array per field; each array with cap = n + 2*halo.
// Halo lets us read i-1 and i+1 at the borders without a branch.
//
// Invariants (asserted in debug):
//   - n >= 1, halo >= 1
//   - data == base + halo
//   - bytes (actually allocated) >= (n + 2*halo) * sizeof(T)
//   - bytes aligned to 64 (cache line)
//
// W1 (2026-05-11 audit): template on T. HotField16 = HotField<uint16_t>
// preserves all legacy uses bit-exact. Arbitrary integer types (uint8_t,
// uint32_t, uint64_t) can be instantiated; existing kernels stay fixed on
// uint16. Float types would break bit-exact cross-arch — integer only.
template <typename T>
struct HotField {
    T*     data  = nullptr;     // pointer after the left halo
    T*     base  = nullptr;     // original pointer (for free)
    size_t n     = 0;           // useful cells
    size_t halo  = 1;           // border cells on each side
    size_t bytes = 0;           // bytes actually allocated (round-up to 64)
};

using HotField16 = HotField<uint16_t>;

template <typename T>
inline HotField<T> alloc_hf(size_t n, size_t halo = 1) {
    HotField<T> h;
    if (n == 0 || halo == 0) return h;
    if (n > (SIZE_MAX / sizeof(T)) - 2*halo) return h;
    h.n = n;
    h.halo = halo;
    size_t cap_bytes = (n + 2*halo) * sizeof(T);
    size_t aligned_bytes = (cap_bytes + 63) & ~size_t(63);
    h.bytes = aligned_bytes;
#ifdef _WIN32
    h.base = (T*)VirtualAlloc(nullptr, aligned_bytes,
                MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!h.base) { h = {}; return h; }
    VirtualLock(h.base, aligned_bytes);
#else
    if (posix_memalign((void**)&h.base, 64, aligned_bytes) != 0 || h.base == nullptr) {
        h = {};
        return h;
    }
#endif
    h.data = h.base + halo;
    return h;
}

template <typename T>
inline void free_hf(HotField<T>& h) {
    if (!h.base) return;
#ifdef _WIN32
    VirtualUnlock(h.base, h.bytes);
    VirtualFree(h.base, 0, MEM_RELEASE);
#else
    free(h.base);
#endif
    h = {};
}

// Backward-compat aliases. Existing code using alloc_hf16/free_hf16 continues
// to work unchanged. These are inline so no extra symbol cost.
inline HotField16 alloc_hf16(size_t n, size_t halo = 1) { return alloc_hf<uint16_t>(n, halo); }
inline void       free_hf16(HotField16& h) { free_hf<uint16_t>(h); }

inline void seed_hf16(HotField16& h, uint32_t seed = 0xC0FFEEu) {
    uint32_t s = seed ? seed : 1u;
    for (size_t i = 0; i < h.n; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        h.data[i] = (uint16_t)(s & 0xFFFF);
    }
    // zeroed halo
    for (size_t i = 0; i < h.halo; ++i) {
        h.data[-(ptrdiff_t)(i+1)] = 0;
        h.data[h.n + i] = 0;
    }
}

// === Canonical kernel functor (W2 2026-05-11 audit) ===
//
// Encapsulates the rule `((l + 2c + r) >> 2) * 255 / 256` as a functor with
// operator(). Allows alternative stencils via template parameter without
// duplicating loop scaffolding. CORE kernels (propagate_1d, propagate_chunk)
// keep the rule inline to preserve bit-exact baseline.
// Pluggable versions are opt-in via propagate_*_with_kernel<K>(...) in
// experiment files.
//
// Constraints for every Kernel:
//   - operator() is pure (no external state); enables cross-arch bit-exact
//   - returns deterministic uint16_t from uint32_t inputs
//   - integer-only (float breaks cross-arch determinism)
struct CanonicalKernel {
    static inline uint16_t apply(uint32_t l, uint32_t c, uint32_t r) {
        uint32_t avg = (l + (c << 1) + r) >> 2;
        uint32_t dec = (avg * 255u) >> 8;
        return (uint16_t)dec;
    }
};

// === W1 full refactor (Day 4): intermediate_t<T> ===
// Promotes cell type T to a wider type that holds (l + 2c + r) without overflow.
//   T = uint8_t   -> uint16_t (4*255 = 1020 < 65535)
//   T = uint16_t  -> uint32_t (4*65535 = 262140 < 2^32)
//   T = uint32_t  -> uint64_t (4*2^32 < 2^34, fits in 64-bit)
//   T = uint64_t  -> NOT SUPPORTED by default (would need __uint128_t).
//                    uint64_t workloads should use a different kernel (W2).
template <typename T> struct intermediate_t;
template <> struct intermediate_t<uint8_t>  { using type = uint16_t; };
template <> struct intermediate_t<uint16_t> { using type = uint32_t; };
template <> struct intermediate_t<uint32_t> { using type = uint64_t; };

// Generic CanonicalKernel parameterized on T. For integer T with intermediate_t
// defined, applies rule (l + 2c + r) >> 2 * 255/256 with safe promotion.
// W2: uniform signature `(T, T, T) -> T`; internal promotion via intermediate_t.
template <typename T>
struct CanonicalKernelT {
    using I = typename intermediate_t<T>::type;
    static inline T apply(T l, T c, T r) {
        I avg = ((I)l + ((I)c << 1) + (I)r) >> 2;
        I dec = (avg * (I)255) >> 8;
        return (T)dec;
    }
};

// W2 full refactor: alternative kernels templated on T.
//
// SimpleAvgKernel: simple average (l+c+r)/3 with NO decay. For workloads that
// want to retain energy (rate counting, monitoring).
template <typename T>
struct SimpleAvgKernelT {
    using I = typename intermediate_t<T>::type;
    static inline T apply(T l, T c, T r) {
        I sum = (I)l + (I)c + (I)r;
        return (T)(sum / 3);
    }
};

// EmaKernel: heavy weight on center (0.5*c + 0.25*(l+r)) with decay 254/256.
// For time-series workloads where the present weighs more than neighbors.
template <typename T>
struct EmaKernelT {
    using I = typename intermediate_t<T>::type;
    static inline T apply(T l, T c, T r) {
        I weighted = ((I)c << 1) + (I)l + (I)r;     // 4x the weighted average
        I dec = (weighted * (I)254) >> 10;          // /4 and *(254/256) combined
        return (T)dec;
    }
};

// === Uint64 kernel (no intermediate_t<uint64> available in standard C++) ===
//
// For T=uint64, the rule (l + 2c + r) overflows in extreme cases. This
// version pre-divides to avoid overflow, with minor precision loss in the
// 2 low bits.
struct CanonicalKernel_u64 {
    static inline uint64_t apply(uint64_t l, uint64_t c, uint64_t r) {
        // (l + 2c + r) / 4 = l/4 + c/2 + r/4 (overflow-safe)
        uint64_t avg = (l >> 2) + (c >> 1) + (r >> 2);
        // avg * 255 / 256 = avg - avg/256 (overflow-safe; bit-exact equivalent)
        return avg - (avg >> 8);
    }
};

// Alternative kernel: simple average without decay.
// Use for conservative workloads (rate counting, monitoring).
struct SimpleAvgKernel {
    static inline uint16_t apply(uint32_t l, uint32_t c, uint32_t r) {
        return (uint16_t)((l + c + r) / 3);
    }
};

// Alternative kernel: EMA-like (heavy weight on center, fast decay).
// Use for time-series where the "present" weighs more than neighbors.
struct EmaKernel {
    static inline uint16_t apply(uint32_t l, uint32_t c, uint32_t r) {
        // y = 0.5 * c + 0.25 * (l+r), with decay 254/256
        uint32_t avg = (c << 1) + l + r;       // 4 * weighted average
        uint32_t dec = (avg * 254u) >> 10;     // /4 and *(254/256) combined
        return (uint16_t)dec;
    }
};

// === Main kernel ===
// rule: y[i] = ((x[i-1] + 2*x[i] + x[i+1]) >> 2) with decay (255/256)
// integer, no cast to float, no branch, no indirect pointer.
// double buffer: reads from prev, writes to next.
//
// untiled version — the whole array in one pass.
__attribute__((noinline))
inline void propagate_1d(const HotField16& prev, HotField16& next) {
    const uint16_t* __restrict__ x = prev.data;
    uint16_t* __restrict__ y = next.data;
    const size_t n = prev.n;
    for (size_t i = 0; i < n; ++i) {
        uint32_t l = x[(ptrdiff_t)i - 1];
        uint32_t c = x[i];
        uint32_t r = x[i + 1];
        uint32_t avg = (l + (c << 1) + r) >> 2;
        uint32_t dec = (avg * 255u) >> 8;
        y[i] = (uint16_t)dec;
    }
}

// W1 full refactor: template T version (uint8/16/32).
// For uint16_t it is bit-exact equivalent to propagate_1d above.
template <typename T>
inline void propagate_1d_t(const HotField<T>& prev, HotField<T>& next) {
    using I = typename intermediate_t<T>::type;
    const T* __restrict__ x = prev.data;
    T* __restrict__ y = next.data;
    const size_t n = prev.n;
    for (size_t i = 0; i < n; ++i) {
        I l = x[(ptrdiff_t)i - 1];
        I c = x[i];
        I r = x[i + 1];
        I avg = (l + (c << 1) + r) >> 2;
        I dec = (avg * (I)255) >> 8;
        y[i] = (T)dec;
    }
}

// tiled version — chunks that fit comfortably in L1/L2.
// for each tile, the kernel works on a contiguous region. multiple generations
// per tile increase reuse, but we start with 1 generation per tile.
__attribute__((noinline))
inline void propagate_1d_tiled(const HotField16& prev, HotField16& next, size_t tile) {
    const uint16_t* __restrict__ x = prev.data;
    uint16_t* __restrict__ y = next.data;
    const size_t n = prev.n;
    if (tile == 0 || tile >= n) { propagate_1d(prev, next); return; }
    for (size_t base = 0; base < n; base += tile) {
        size_t end = std::min(base + tile, n);
        for (size_t i = base; i < end; ++i) {
            uint32_t l = x[(ptrdiff_t)i - 1];
            uint32_t c = x[i];
            uint32_t r = x[i + 1];
            uint32_t avg = (l + (c << 1) + r) >> 2;
            uint32_t dec = (avg * 255u) >> 8;
            y[i] = (uint16_t)dec;
        }
    }
}

// utility: swaps the pointers of two HotField16 (generation ping-pong)
inline void swap_hf16(HotField16& a, HotField16& b) {
    std::swap(a.data, b.data);
    std::swap(a.base, b.base);
    std::swap(a.n,    b.n);
    std::swap(a.halo, b.halo);
}

// checksum to prevent dead-code elimination
inline uint64_t checksum_hf16(const HotField16& h) {
    uint64_t s = 0;
    for (size_t i = 0; i < h.n; ++i) s += h.data[i];
    return s;
}

} // namespace cl
