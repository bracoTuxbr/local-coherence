// tissue2d.hpp — linearized 2D tissue, perimeter halo
//
// Layout: row-major, dimensions (H, W). Halo of 1 cell on each border
// (top, bottom, left, right). data[] points to the inner (0,0) cell,
// indexable as data[r * stride + c]. base[] is the original pointer (with halo).
// stride = W + 2*halo (optional 64-byte alignment via padding).

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

namespace cl {

struct HotField2D {
    uint16_t* data   = nullptr;   // inner (0,0); use [r * stride + c]
    uint16_t* base   = nullptr;
    size_t    H      = 0;
    size_t    W      = 0;
    size_t    stride = 0;         // = W + 2*halo
    size_t    halo   = 1;
    size_t    bytes  = 0;
};

// alloc_hf2d — allocates a 2D tissue (H x W cells) with perimeter halo.
// On failure, returns a struct with base=nullptr. Caller MUST check.
inline HotField2D alloc_hf2d(size_t H, size_t W, size_t halo = 1) {
    HotField2D f;
    if (H == 0 || W == 0 || halo == 0) return f;
    // overflow checks
    if (H > SIZE_MAX / (W + 2*halo + 1)) return f;
    f.H = H; f.W = W; f.halo = halo;
    f.stride = W + 2 * halo;
    size_t total = (H + 2*halo) * f.stride * sizeof(uint16_t);
    size_t aligned = (total + 4095) & ~size_t(4095);
    f.bytes = aligned;
#ifdef _WIN32
    f.base = (uint16_t*)VirtualAlloc(nullptr, aligned,
        MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!f.base) { f = {}; return f; }
    VirtualLock(f.base, aligned);
#else
    if (posix_memalign((void**)&f.base, 64, aligned) != 0 || f.base == nullptr) {
        f = {};
        return f;
    }
#endif
    // zero everything
    for (size_t i = 0; i < aligned / sizeof(uint16_t); ++i) f.base[i] = 0;
    // data points to (halo, halo)
    f.data = f.base + (halo * f.stride) + halo;
    return f;
}

inline void free_hf2d(HotField2D& f) {
    if (!f.base) return;
#ifdef _WIN32
    VirtualUnlock(f.base, f.bytes);
    VirtualFree(f.base, 0, MEM_RELEASE);
#else
    free(f.base);
#endif
    f = {};
}

inline void clear_hf2d(HotField2D& f) {
    for (size_t r = 0; r < f.H; ++r) {
        uint16_t* row = f.data + r * f.stride;
        for (size_t c = 0; c < f.W; ++c) row[c] = 0;
    }
}

inline void set_cell(HotField2D& f, size_t r, size_t c, uint16_t v) {
    f.data[r * f.stride + c] = v;
}
inline uint16_t get_cell(const HotField2D& f, size_t r, size_t c) {
    return f.data[r * f.stride + c];
}

// sum over the whole grid (ignores halo) — for checks
inline uint64_t sum_hf2d(const HotField2D& f) {
    uint64_t s = 0;
    for (size_t r = 0; r < f.H; ++r) {
        const uint16_t* row = f.data + r * f.stride;
        for (size_t c = 0; c < f.W; ++c) s += row[c];
    }
    return s;
}

// L1 distance between two HotField2D (for correctness)
inline uint64_t l1_hf2d(const HotField2D& A, const HotField2D& B) {
    uint64_t s = 0;
    size_t H = A.H < B.H ? A.H : B.H;
    size_t W = A.W < B.W ? A.W : B.W;
    for (size_t r = 0; r < H; ++r) {
        const uint16_t* ra = A.data + r * A.stride;
        const uint16_t* rb = B.data + r * B.stride;
        for (size_t c = 0; c < W; ++c) {
            int da = ra[c], db = rb[c];
            s += (uint64_t)(da > db ? da - db : db - da);
        }
    }
    return s;
}

} // namespace cl
