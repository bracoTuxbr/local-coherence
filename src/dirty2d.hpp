// dirty2d.hpp — 2D tile bitmap + stability + 4-connected propagation
//
// Tile = TILE_R x TILE_C cells (default 64x64 = 4096 cells = 8 KB).
// Bitmap: 1 bit per tile. Linearized in uint64_t[]. The tile grid has
// dimensions (NTR x NTC). If NTR*NTC < 64*words, leftover bits go unused.
//
// When a tile-border cell changes >= eps, propagates dirty to the neighbor
// tile in the border direction (4-connected: N, S, E, W).

#pragma once

#include "tissue2d.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <cstdio>

namespace cl {

constexpr size_t TILE_R              = 64;
constexpr size_t TILE_C              = 64;
constexpr uint16_t SIG_DELTA_2D      = 4;
constexpr uint8_t  STABLE_THRESH_2D  = 8;

struct DirtyTissue2D {
    HotField2D a;
    HotField2D b;
    uint8_t*   stability = nullptr;   // counter[H * W] (no stride; internal only)
    uint64_t*  dirty     = nullptr;
    size_t     H = 0, W = 0;
    size_t     NTR = 0, NTC = 0;      // # tiles in each dimension
    size_t     n_tiles = 0;
    size_t     n_words = 0;
    size_t     stab_stride = 0;       // = W; row-major, no halo
};

// alloc_dirty2d — allocates 2D tissues a/b + stability + dirty bitmap.
// On any failure, frees everything and returns an empty struct.
// Caller MUST check d.a.base != nullptr before use.
inline DirtyTissue2D alloc_dirty2d(size_t H, size_t W) {
    DirtyTissue2D d;
    if (H == 0 || W == 0) return d;
    // overflow check on H*W
    if (H > SIZE_MAX / W) return d;
    d.H = H; d.W = W;
    d.NTR = (H + TILE_R - 1) / TILE_R;
    d.NTC = (W + TILE_C - 1) / TILE_C;
    d.n_tiles = d.NTR * d.NTC;
    d.n_words = (d.n_tiles + 63) / 64;
    d.stab_stride = W;

    d.a = alloc_hf2d(H, W, /*halo=*/1);
    if (!d.a.base) { d = {}; return d; }
    d.b = alloc_hf2d(H, W, /*halo=*/1);
    if (!d.b.base) { free_hf2d(d.a); d = {}; return d; }

    size_t stab_bytes = ((H*W) + 63) & ~size_t(63);
    size_t dirty_bytes = d.n_words * sizeof(uint64_t);
#ifdef _WIN32
    d.stability = (uint8_t*)VirtualAlloc(nullptr, stab_bytes,
        MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    d.dirty = (uint64_t*)VirtualAlloc(nullptr, dirty_bytes,
        MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    if (posix_memalign((void**)&d.stability, 64, stab_bytes) != 0) d.stability = nullptr;
    if (posix_memalign((void**)&d.dirty,     64, dirty_bytes) != 0) d.dirty = nullptr;
#endif
    if (!d.stability || !d.dirty) {
        free_hf2d(d.a); free_hf2d(d.b);
#ifdef _WIN32
        if (d.stability) VirtualFree(d.stability, 0, MEM_RELEASE);
        if (d.dirty)     VirtualFree(d.dirty, 0, MEM_RELEASE);
#else
        free(d.stability); free(d.dirty);
#endif
        d = {};
        return d;
    }
    std::memset(d.stability, 0, H*W);
    std::memset(d.dirty, 0xFF, dirty_bytes);
    return d;
}

inline void free_dirty2d(DirtyTissue2D& d) {
    free_hf2d(d.a);
    free_hf2d(d.b);
#ifdef _WIN32
    if (d.stability) VirtualFree(d.stability, 0, MEM_RELEASE);
    if (d.dirty)     VirtualFree(d.dirty, 0, MEM_RELEASE);
#else
    free(d.stability); free(d.dirty);
#endif
    d = {};
}

// === bitmap helpers (linearized as tile_idx = tr * NTC + tc) ===
static inline size_t tile_idx(const DirtyTissue2D& d, size_t tr, size_t tc) {
    return tr * d.NTC + tc;
}
static inline bool tile_is_dirty(const uint64_t* bits, size_t idx) {
    return (bits[idx / 64] & (1ULL << (idx % 64))) != 0;
}
static inline void tile_set_dirty(uint64_t* bits, size_t idx) {
    bits[idx / 64] |= (1ULL << (idx % 64));
}

inline size_t count_dirty_tiles(const DirtyTissue2D& d, const uint64_t* bits) {
    size_t cnt = 0;
    for (size_t t = 0; t < d.n_tiles; ++t)
        if (tile_is_dirty(bits, t)) ++cnt;
    return cnt;
}

// injector — perturbs a rectangular region [r0, r1) x [c0, c1) in prev
inline void inject_pulse_2d(DirtyTissue2D& d, bool prev_is_a,
                            size_t r0, size_t r1, size_t c0, size_t c1,
                            uint16_t value)
{
    HotField2D& prev = prev_is_a ? d.a : d.b;
    if (r1 > d.H) r1 = d.H;
    if (c1 > d.W) c1 = d.W;
    for (size_t r = r0; r < r1; ++r) {
        uint16_t* row = prev.data + r * prev.stride;
        for (size_t c = c0; c < c1; ++c) {
            row[c] = value;
            d.stability[r * d.stab_stride + c] = 0;
        }
    }
    size_t tr0 = r0 / TILE_R, tr1 = (r1 + TILE_R - 1) / TILE_R;
    size_t tc0 = c0 / TILE_C, tc1 = (c1 + TILE_C - 1) / TILE_C;
    for (size_t tr = tr0; tr < tr1; ++tr)
        for (size_t tc = tc0; tc < tc1; ++tc)
            tile_set_dirty(d.dirty, tile_idx(d, tr, tc));
    // also dirty the neighbor tiles (perturbation will propagate)
    if (tr0 > 0) for (size_t tc = tc0; tc < tc1; ++tc)
        tile_set_dirty(d.dirty, tile_idx(d, tr0-1, tc));
    if (tr1 < d.NTR) for (size_t tc = tc0; tc < tc1; ++tc)
        tile_set_dirty(d.dirty, tile_idx(d, tr1, tc));
    if (tc0 > 0) for (size_t tr = tr0; tr < tr1; ++tr)
        tile_set_dirty(d.dirty, tile_idx(d, tr, tc0-1));
    if (tc1 < d.NTC) for (size_t tr = tr0; tr < tr1; ++tr)
        tile_set_dirty(d.dirty, tile_idx(d, tr, tc1));
}

} // namespace cl
