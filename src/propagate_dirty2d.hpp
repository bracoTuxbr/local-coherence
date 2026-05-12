// propagate_dirty2d.hpp — 2D kernel with tile skip + freeze + 4-connected propagation
//
// For each dirty tile:
//   - process every cell in the tile (TILE_R x TILE_C, with border clamping)
//   - update per-cell stability counter
//   - track whether border N/S/E/W changed >= eps; if so, propagate dirty to neighbor
//   - if no cell changed >= eps, tile leaves the queue (no bit set in next_dirty)

#pragma once

#include "tissue2d.hpp"
#include "dirty2d.hpp"
#include "diffuse2d.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace cl {

// returns by reference: chunk_changed and border flags
__attribute__((noinline))
inline bool propagate_tile_2d(const HotField2D& prev, HotField2D& next,
                              uint8_t* stab, size_t stab_stride,
                              size_t tr, size_t tc,
                              size_t H, size_t W,
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

            if (dlt < SIG_DELTA_2D) {
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

// full pass over the bitmap. returns number of processed tiles.
__attribute__((noinline))
inline size_t propagate_dirty2d_pass(DirtyTissue2D& d, bool prev_is_a,
                                     uint64_t* next_dirty)
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
            bool changed = propagate_tile_2d(prev, next, d.stability, d.stab_stride,
                tr, tc, d.H, d.W, tn, ts, tw, te);
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

inline void apply_next_dirty2d(DirtyTissue2D& d, const uint64_t* next_dirty) {
    std::memcpy(d.dirty, next_dirty, d.n_words * sizeof(uint64_t));
}

} // namespace cl
