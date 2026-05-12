// propagate_dirty2d_adaptive.hpp — 2D dirty kernel variant with a per-local-
// magnitude adaptive threshold. A fixed SIG_DELTA was fine for synthetic
// signals near zero; on real FFT output (mel-spec uint16 with magnitude ~16k
// even during silence), an absolute threshold never triggers freeze in
// reasonable time.
//
// Solution: delta_threshold[i] = max(MIN_SIG_DELTA, value[i] >> SHIFT).
// For SHIFT=8, value 16k -> threshold 64; value 100 -> threshold 0 (clamped).
// High-magnitude cells need proportional change to "count".
//
// Reuses DirtyTissue2D from dirty2d.hpp. Pass SHIFT as a kernel parameter.

#pragma once

#include "tissue2d.hpp"
#include "dirty2d.hpp"
#include "diffuse2d.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace cl {

constexpr uint16_t MIN_SIG_DELTA   = 4;
constexpr int      DEFAULT_SHIFT   = 8;

// single-tile kernel with adaptive threshold
__attribute__((noinline))
inline bool propagate_tile_2d_adaptive(
    const HotField2D& prev, HotField2D& next,
    uint8_t* stab, size_t stab_stride,
    size_t tr, size_t tc,
    size_t H, size_t W, int shift,
    bool& touched_north, bool& touched_south,
    bool& touched_west,  bool& touched_east)
{
    size_t r0 = tr * TILE_R;
    size_t r1 = std::min(r0 + TILE_R, H);
    size_t c0 = tc * TILE_C;
    size_t c1 = std::min(c0 + TILE_C, W);

    const uint16_t* __restrict__ x = prev.data;
    uint16_t* __restrict__ y = next.data;
    const size_t s = prev.stride;

    bool changed = false;
    touched_north = touched_south = touched_west = touched_east = false;

    for (size_t r = r0; r < r1; ++r) {
        const uint16_t* row    = x + r * s;
        const uint16_t* row_up = x + (r - 1) * s;
        const uint16_t* row_dn = x + (r + 1) * s;
        uint16_t* yrow = y + r * s;
        uint8_t*  srow = stab + r * stab_stride;
        for (size_t c = c0; c < c1; ++c) {
            uint32_t up = row_up[c];
            uint32_t dn = row_dn[c];
            uint32_t lf = row[c - 1];
            uint32_t rt = row[c + 1];
            uint32_t cn = row[c];
            uint32_t avg = (up + dn + lf + rt + (cn << 2)) >> 3;
            uint32_t dec = (avg * 255u) >> 8;
            uint16_t newv = (uint16_t)dec;
            uint16_t oldv = (uint16_t)cn;
            uint16_t dlt  = newv > oldv ? newv - oldv : oldv - newv;

            // adaptive threshold: max(MIN, max(newv, oldv) >> shift)
            uint32_t mag = newv > oldv ? newv : oldv;
            uint32_t thr = mag >> (uint32_t)shift;
            if (thr < MIN_SIG_DELTA) thr = MIN_SIG_DELTA;

            if (dlt < thr) {
                if (srow[c] < 0xFF) ++srow[c];
            } else {
                srow[c] = 0;
                changed = true;
                if (r == r0)        touched_north = true;
                if (r == r1 - 1)    touched_south = true;
                if (c == c0)        touched_west  = true;
                if (c == c1 - 1)    touched_east  = true;
            }
            yrow[c] = newv;
        }
    }
    return changed;
}

// full pass with adaptive threshold
__attribute__((noinline))
inline size_t propagate_dirty2d_adaptive_pass(DirtyTissue2D& d, bool prev_is_a,
                                              uint64_t* next_dirty, int shift)
{
    HotField2D& prev = prev_is_a ? d.a : d.b;
    HotField2D& next = prev_is_a ? d.b : d.a;
    std::memset(next_dirty, 0, d.n_words * sizeof(uint64_t));
    size_t processed = 0;

    for (size_t w = 0; w < d.n_words; ++w) {
        uint64_t word = d.dirty[w];
        if (word == 0) continue;
        while (word) {
            int b = __builtin_ctzll(word);
            word &= word - 1;
            size_t idx = w * 64 + (size_t)b;
            if (idx >= d.n_tiles) break;
            size_t tr = idx / d.NTC;
            size_t tc = idx % d.NTC;

            bool tn=false, ts=false, tw=false, te=false;
            bool changed = propagate_tile_2d_adaptive(prev, next, d.stability,
                d.stab_stride, tr, tc, d.H, d.W, shift, tn, ts, tw, te);
            ++processed;

            if (changed) {
                tile_set_dirty(next_dirty, idx);
                if (tn && tr > 0)
                    tile_set_dirty(next_dirty, tile_idx(d, tr-1, tc));
                if (ts && tr+1 < d.NTR)
                    tile_set_dirty(next_dirty, tile_idx(d, tr+1, tc));
                if (tw && tc > 0)
                    tile_set_dirty(next_dirty, tile_idx(d, tr, tc-1));
                if (te && tc+1 < d.NTC)
                    tile_set_dirty(next_dirty, tile_idx(d, tr, tc+1));
            }
        }
    }
    return processed;
}

} // namespace cl
