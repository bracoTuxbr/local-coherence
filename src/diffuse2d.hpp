// diffuse2d.hpp — 2D diffusion kernel (5-point stencil), naive and tiled
//
// Rule: y[r,c] = ((x[r-1,c] + x[r+1,c] + x[r,c-1] + x[r,c+1] + 4*x[r,c]) >> 3) * 255 / 256
// (simple 5-point average with weight 4 at the center; decay each gen)
//
// The rule is symmetric and deterministically convergent; serves as 2D analog
// to the 1D (l + 2c + r) >> 2 from M1.
#pragma once

#include "tissue2d.hpp"
#include <cstdint>
#include <algorithm>

namespace cl {

// naive kernel: applies 1 generation across the whole tissue. prev -> next.
// halo must be zeroed (Dirichlet) or set by the caller before the call.
__attribute__((noinline))
inline void diffuse2d_pass(const HotField2D& prev, HotField2D& next) {
    const size_t H = prev.H;
    const size_t W = prev.W;
    const size_t s = prev.stride;
    const uint16_t* __restrict__ x = prev.data;
    uint16_t* __restrict__ y = next.data;
    for (size_t r = 0; r < H; ++r) {
        const uint16_t* row    = x + r * s;
        const uint16_t* row_up = x + (r - 1) * s; // halo absorbs r=0
        const uint16_t* row_dn = x + (r + 1) * s;
        uint16_t* yrow = y + r * s;
        for (size_t c = 0; c < W; ++c) {
            uint32_t up = row_up[c];
            uint32_t dn = row_dn[c];
            uint32_t lf = row[c - 1]; // halo absorbs c=0
            uint32_t rt = row[c + 1];
            uint32_t cn = row[c];
            uint32_t avg = (up + dn + lf + rt + (cn << 2)) >> 3;
            uint32_t dec = (avg * 255u) >> 8;
            yrow[c] = (uint16_t)dec;
        }
    }
}

// tiled kernel in blocks (BR x BC), single-thread for now.
// used to confirm the tile-blocking effect in 2D before dirty.
__attribute__((noinline))
inline void diffuse2d_tiled(const HotField2D& prev, HotField2D& next,
                            size_t BR, size_t BC) {
    const size_t H = prev.H;
    const size_t W = prev.W;
    if (BR == 0) BR = H;
    if (BC == 0) BC = W;
    for (size_t br = 0; br < H; br += BR) {
        for (size_t bc = 0; bc < W; bc += BC) {
            size_t er = std::min(br + BR, H);
            size_t ec = std::min(bc + BC, W);
            const size_t s = prev.stride;
            const uint16_t* __restrict__ x = prev.data;
            uint16_t* __restrict__ y = next.data;
            for (size_t r = br; r < er; ++r) {
                const uint16_t* row    = x + r * s;
                const uint16_t* row_up = x + (r - 1) * s;
                const uint16_t* row_dn = x + (r + 1) * s;
                uint16_t* yrow = y + r * s;
                for (size_t c = bc; c < ec; ++c) {
                    uint32_t up = row_up[c];
                    uint32_t dn = row_dn[c];
                    uint32_t lf = row[c - 1];
                    uint32_t rt = row[c + 1];
                    uint32_t cn = row[c];
                    uint32_t avg = (up + dn + lf + rt + (cn << 2)) >> 3;
                    uint32_t dec = (avg * 255u) >> 8;
                    yrow[c] = (uint16_t)dec;
                }
            }
        }
    }
}

} // namespace cl
