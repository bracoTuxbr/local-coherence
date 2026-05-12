// propagate_temporal.hpp — temporal tile blocking (trapezoidal)
//
// ============================================================================
//  CORE PARADIGM FILE (M2.5).
//  K-step trapezoid ping-pong in local buffer is canonical (94% efficiency at 32MB).
//  Verify modifications via tools/regression_test.ps1 against
//  benchmarks/golden_numbers.txt.
// ============================================================================
//
// Principle: each tile [s, e) loads [s-K, e+K) from the global buffer into a
// small local buffer (fits in L1/L2). Runs K generations locally, ping-ponging
// between 2 local buffers. Halo shrinks by 1 per generation. After K, writes
// [s, e) into the global output buffer.
//
// Total DRAM traffic / K. (Only 1 read and 1 write per cell per K-block.)
//
// Trade-off: redundant computation at inter-thread borders (each thread
// recomputes K cells on each side per K-block). For tile >> K, it is negligible.

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>

#include "propagate.hpp"  // HotField16, base rule

namespace cl {

// Maximum local buffer size in bytes (fits comfortably in L1 32KB if K and
// tile are moderate; fits in L2 512KB for large K and tile).
// Each temporal kernel call allocates TWO buffers of this size.
constexpr size_t TEMPORAL_LOCAL_MAX_CELLS = 65536; // 128 KB per buffer

// Main kernel: applies K generations to [s, e) using a local buffer of
// size W = (e - s) + 2K.
//
// Precondition: W <= TEMPORAL_LOCAL_MAX_CELLS.
// Precondition: prev has valid data in [s-K, e+K).
__attribute__((noinline))
inline void propagate_temporal_tile(
    const HotField16& prev, HotField16& next,
    size_t s, size_t e, int K,
    uint16_t* L0, uint16_t* L1)
{
    const size_t W = e - s;
    const size_t cap = W + 2 * (size_t)K;

    // 1) copy [s-K, e+K) from prev into L0.
    //    local indexing: L0[i] corresponds to prev.data[(s-K) + i]
    const uint16_t* src = prev.data + (ptrdiff_t)(s) - (ptrdiff_t)K;
    std::memcpy(L0, src, cap * sizeof(uint16_t));

    uint16_t* in_buf  = L0;
    uint16_t* out_buf = L1;

    // 2) K generations in the local buffer. Valid range shrinks by 1 per generation:
    //    generation k computes indices [k+1, cap - k - 1) of in_buf -> out_buf
    for (int k = 0; k < K; ++k) {
        size_t valid_start = (size_t)k + 1;
        size_t valid_end   = cap - (size_t)k - 1;
        // local kernel identical to the global one
        for (size_t i = valid_start; i < valid_end; ++i) {
            uint32_t l = in_buf[i - 1];
            uint32_t c = in_buf[i];
            uint32_t r = in_buf[i + 1];
            uint32_t avg = (l + (c << 1) + r) >> 2;
            uint32_t dec = (avg * 255u) >> 8;
            out_buf[i] = (uint16_t)dec;
        }
        std::swap(in_buf, out_buf);
    }
    // after K iterations, the result is in in_buf (after the final swap)

    // 3) copy the useful window [K, K+W) of the result buffer into next.data[s..e)
    std::memcpy(next.data + s, in_buf + K, W * sizeof(uint16_t));
}

// Wrapper to sweep a thread range in sub-tiles.
// Pre: range [t_start, t_end) is the thread's "territory" (already without halo).
// Pre: K-step block — all threads call this before the global barrier.
inline void propagate_temporal_range(
    const HotField16& prev, HotField16& next,
    size_t t_start, size_t t_end, size_t tile_W, int K,
    uint16_t* L0, uint16_t* L1)
{
    if (tile_W == 0) tile_W = (t_end - t_start);
    for (size_t s = t_start; s < t_end; s += tile_W) {
        size_t e = std::min(s + tile_W, t_end);
        propagate_temporal_tile(prev, next, s, e, K, L0, L1);
    }
}

} // namespace cl
