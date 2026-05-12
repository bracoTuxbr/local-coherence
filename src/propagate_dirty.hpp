// propagate_dirty.hpp — kernel with chunk skip + freeze by stability
//
// ============================================================================
//  CORE PARADIGM FILE (M4).
//  Dirty bitmap + freeze + cross-chunk reactivation is canonical (626x speedup
//  on pulse 1D). L1_distance=94148 with the standard pulse is EXACT bit-exact
//  and must be preserved across any modification (verify via
//  tools/regression_test.ps1 + benchmarks/golden_numbers.txt).
// ============================================================================
//
// For each dirty chunk (64 cells):
//   - process the 64 cells: compute new = stencil(prev). If a cell is frozen
//     (stability >= STABLE_THRESH), check whether neighbor perturbation in the
//     chunk justifies reactivating: |delta_in| > HYSTERESIS_DELTA -> reset.
//   - keep a chunk_changed flag that becomes true if ANY cell changed >= eps.
//   - if chunk_changed: rewrite dirty bit (stays set); if a chunk border
//     changed, propagate dirty to the neighbor chunk.
//   - if none changed: clear this chunk's bit (CPU drops).
//
// Single-thread for now. Multi-thread comes in e06 if M4 passes.

#pragma once

#include "dirty.hpp"
#include "propagate.hpp"

#include <cstdint>
#include <cstdlib>
#include <algorithm>

namespace cl {

// W1 full refactor: template T version (uint8/16/32, or uint64 with a custom Kernel).
// W2 full refactor: now takes a Kernel template parameter; default = CanonicalKernelT<T>.
// Kernel must have `static T apply(T l, T c, T r)` — internal promotion is the
// Kernel's responsibility. CanonicalKernelT<T> uses intermediate_t<T>;
// CanonicalKernel_u64 uses overflow-safe arithmetic.
template <typename T, typename Kernel = CanonicalKernelT<T>>
__attribute__((noinline))
inline bool propagate_chunk_t(const HotField<T>& prev, HotField<T>& next,
                              uint8_t* stability,
                              size_t chunk_idx, size_t n,
                              T sig_delta = (T)SIGNIFICANT_DELTA)
{
    size_t s = chunk_idx * CHUNK_CELLS;
    size_t e = std::min(s + CHUNK_CELLS, n);
    const T* __restrict__ x = prev.data;
    T* __restrict__ y = next.data;

    bool changed = false;
    bool first_changed = false;
    bool last_changed  = false;

    for (size_t i = s; i < e; ++i) {
        T l = x[(ptrdiff_t)i - 1];
        T c = x[i];
        T r = x[i + 1];
        T newv = Kernel::apply(l, c, r);
        T oldv = c;
        T delta = (newv > oldv) ? (T)(newv - oldv) : (T)(oldv - newv);

        if (delta < sig_delta) {
            if (stability[i] < 0xFF) ++stability[i];
        } else {
            stability[i] = 0;
            changed = true;
            if (i == s)         first_changed = true;
            if (i == e - 1)     last_changed  = true;
        }
        y[i] = newv;
    }
    (void)first_changed; (void)last_changed;
    return changed;
}

// returns true if chunk_changed (any cell changed >= eps in this chunk)
// sig_delta: runtime-tunable threshold (W3); default keeps canonical behavior.
__attribute__((noinline))
inline bool propagate_chunk(const HotField16& prev, HotField16& next,
                            uint8_t* stability,
                            size_t chunk_idx, size_t n,
                            uint16_t sig_delta = SIGNIFICANT_DELTA)
{
    size_t s = chunk_idx * CHUNK_CELLS;
    size_t e = std::min(s + CHUNK_CELLS, n);
    const uint16_t* __restrict__ x = prev.data;
    uint16_t* __restrict__ y = next.data;

    bool changed = false;
    bool first_changed = false;
    bool last_changed  = false;

    for (size_t i = s; i < e; ++i) {
        // if the cell is frozen, we still compute (could be expensive).
        // but: if ALL neighbors are also frozen, the input is stable,
        // and new = old. Full chunk skipping depends on that.
        // simplicity: we always compute cells in dirty chunks.
        uint32_t l = x[(ptrdiff_t)i - 1];
        uint32_t c = x[i];
        uint32_t r = x[i + 1];
        uint32_t avg = (l + (c << 1) + r) >> 2;
        uint32_t dec = (avg * 255u) >> 8;
        uint16_t newv = (uint16_t)dec;
        uint16_t oldv = (uint16_t)c;
        uint16_t delta = (newv > oldv) ? (newv - oldv) : (oldv - newv);

        if (delta < sig_delta) {
            // stabilizing on this tick
            if (stability[i] < 0xFF) ++stability[i];
        } else {
            // actually moved
            stability[i] = 0;
            changed = true;
            if (i == s)         first_changed = true;
            if (i == e - 1)     last_changed  = true;
        }
        y[i] = newv;
    }
    (void)first_changed; (void)last_changed; // used by caller
    return changed;
}

// "dirty-aware" version: scans bitmap in 64-bit words, skips zeroed chunks.
// returns number of processed chunks (measure of effective CPU%).
__attribute__((noinline))
inline size_t propagate_dirty_pass(DirtyTissue& d,
                                   bool prev_is_a,
                                   uint64_t* next_dirty_bits)
{
    HotField16& prev = prev_is_a ? d.a : d.b;
    HotField16& next = prev_is_a ? d.b : d.a;
    size_t processed = 0;
    const size_t n_chunks = d.n_chunks;

    // clear output bitmap — we only set bits for chunks that stay dirty
    // or are freshly perturbed by neighbors
    std::memset(next_dirty_bits, 0, d.n_words * sizeof(uint64_t));

    for (size_t w = 0; w < d.n_words; ++w) {
        uint64_t word = d.dirty[w];
        if (word == 0) continue;
        // iterate set bits via __builtin_ctzll
        while (word) {
            int b = __builtin_ctzll(word);
            word &= word - 1;
            size_t chunk = w * 64 + (size_t)b;
            if (chunk >= n_chunks) break;

            // PREFETCH (Sprint v0 #60): on sparse workloads, dirty chunks
            // are cache-cold because they are scattered. While processing
            // the current chunk, hint the hardware prefetcher to fetch the
            // cells of the NEXT dirty chunk (we already know which one via
            // ctz of the residual word). Locality hint = 3 (high temporal:
            // we will use it soon).
            if (word) {
                int next_b = __builtin_ctzll(word);
                size_t next_chunk = w * 64 + (size_t)next_b;
                size_t next_cs = next_chunk * CHUNK_CELLS;
                if (next_cs < d.n) {
                    __builtin_prefetch(prev.data + next_cs, 0, 3);
                }
            }

            // to detect a changed border, we need to know which cell
            // triggered chunk_changed. simple: redo the border check.
            size_t cs = chunk * CHUNK_CELLS;
            size_t ce = std::min(cs + CHUNK_CELLS, d.n);

            bool changed = propagate_chunk(prev, next, d.stability, chunk, d.n, d.sig_delta);
            ++processed;

            if (changed) {
                next_dirty_bits[chunk / 64] |= (1ULL << (chunk % 64));
                // border: if the first cell of the chunk changed, dirty chunk-1
                {
                    uint16_t newv = next.data[cs];
                    uint16_t oldv = prev.data[cs];
                    uint16_t dlt  = newv > oldv ? newv - oldv : oldv - newv;
                    if (dlt >= d.sig_delta && chunk > 0) {
                        next_dirty_bits[(chunk-1) / 64] |= (1ULL << ((chunk-1) % 64));
                    }
                }
                // if the last cell of the chunk changed, dirty chunk+1
                {
                    size_t li = ce - 1;
                    uint16_t newv = next.data[li];
                    uint16_t oldv = prev.data[li];
                    uint16_t dlt  = newv > oldv ? newv - oldv : oldv - newv;
                    if (dlt >= d.sig_delta && chunk + 1 < n_chunks) {
                        next_dirty_bits[(chunk+1) / 64] |= (1ULL << ((chunk+1) % 64));
                    }
                }
            }
            // if !changed: chunk_clear (we do not set in next_dirty_bits)
            // and cells increment stability — eventually all frozen
        }
    }
    return processed;
}

// erases the current bitmap and copies "next" (pre-computed) into current
inline void apply_next_dirty(DirtyTissue& d, const uint64_t* next_dirty_bits) {
    std::memcpy(d.dirty, next_dirty_bits, d.n_words * sizeof(uint64_t));
}

// W1 full refactor: template T version.
template <typename T>
inline void apply_next_dirty_t(DirtyTissue_t<T>& d, const uint64_t* next_dirty_bits) {
    std::memcpy(d.dirty, next_dirty_bits, d.n_words * sizeof(uint64_t));
}

// W1 full refactor: propagate_dirty_pass template on T.
// W2 full refactor: takes a Kernel template parameter; default = CanonicalKernelT<T>.
template <typename T, typename Kernel = CanonicalKernelT<T>>
__attribute__((noinline))
inline size_t propagate_dirty_pass_t(DirtyTissue_t<T>& d,
                                     bool prev_is_a,
                                     uint64_t* next_dirty_bits)
{
    HotField<T>& prev = prev_is_a ? d.a : d.b;
    HotField<T>& next = prev_is_a ? d.b : d.a;
    size_t processed = 0;
    const size_t n_chunks = d.n_chunks;
    std::memset(next_dirty_bits, 0, d.n_words * sizeof(uint64_t));

    for (size_t w = 0; w < d.n_words; ++w) {
        uint64_t word = d.dirty[w];
        if (word == 0) continue;
        while (word) {
            int b = __builtin_ctzll(word);
            word &= word - 1;
            size_t chunk = w * 64 + (size_t)b;
            if (chunk >= n_chunks) break;
            if (word) {
                int next_b = __builtin_ctzll(word);
                size_t next_chunk = w * 64 + (size_t)next_b;
                size_t next_cs = next_chunk * CHUNK_CELLS;
                if (next_cs < d.n) {
                    __builtin_prefetch(prev.data + next_cs, 0, 3);
                }
            }
            size_t cs = chunk * CHUNK_CELLS;
            size_t ce = std::min(cs + CHUNK_CELLS, d.n);
            bool changed = propagate_chunk_t<T, Kernel>(prev, next, d.stability,
                                                        chunk, d.n, d.sig_delta);
            ++processed;
            if (changed) {
                next_dirty_bits[chunk / 64] |= (1ULL << (chunk % 64));
                {
                    T newv = next.data[cs];
                    T oldv = prev.data[cs];
                    T dlt  = newv > oldv ? (T)(newv - oldv) : (T)(oldv - newv);
                    if (dlt >= d.sig_delta && chunk > 0) {
                        next_dirty_bits[(chunk-1) / 64] |= (1ULL << ((chunk-1) % 64));
                    }
                }
                {
                    size_t li = ce - 1;
                    T newv = next.data[li];
                    T oldv = prev.data[li];
                    T dlt  = newv > oldv ? (T)(newv - oldv) : (T)(oldv - newv);
                    if (dlt >= d.sig_delta && chunk + 1 < n_chunks) {
                        next_dirty_bits[(chunk+1) / 64] |= (1ULL << ((chunk+1) % 64));
                    }
                }
            }
        }
    }
    return processed;
}

// pulse injector — perturbs range [lo, hi) in prev (without touching next),
// forces corresponding chunks to dirty
inline void inject_pulse(DirtyTissue& d, bool prev_is_a,
                         size_t lo, size_t hi, uint16_t value) {
    HotField16& prev = prev_is_a ? d.a : d.b;
    if (hi > d.n) hi = d.n;
    if (lo > hi) lo = hi;
    for (size_t i = lo; i < hi; ++i) {
        prev.data[i] = value;
        d.stability[i] = 0;
    }
    size_t cs = lo / CHUNK_CELLS;
    size_t ce = (hi + CHUNK_CELLS - 1) / CHUNK_CELLS;
    for (size_t c = cs; c < ce; ++c) chunk_set_dirty(d.dirty, c);
    // also dirty the immediate neighbor chunks (perturbation will propagate to them)
    if (cs > 0)             chunk_set_dirty(d.dirty, cs - 1);
    if (ce < d.n_chunks)    chunk_set_dirty(d.dirty, ce);
}

} // namespace cl
