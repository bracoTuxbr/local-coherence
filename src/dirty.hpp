// dirty.hpp — chunked bitmap + stability counter
//
// Model: the tissue is split into 64-cell chunks. Each chunk has 1 bit in the
// dirty bitmap. Worker walks the bitmap in 64-bit words and skips zeroed chunks.
// Cells inside a dirty chunk are processed; each cell carries a stability
// counter (uint8). When the counter hits STABLE_THRESH, the cell becomes
// "frozen" — does not recompute. Frozen is encoded in the counter, not in a
// separate bit, reducing footprint.
//
// Criteria:
//   - SIGNIFICANT_DELTA = 4   — diff < 4 counts as "no change"
//   - STABLE_THRESH      = 8  — 8 consecutive ticks without change => frozen
//   - HYSTERESIS_DELTA   = 16 — only leaves frozen if neighbor perturbs > 16
//
// Cross-thread sync: chunk_size = 64 cells * 2 bytes = 128 B (2 cache lines).
// Partitioning by aligned chunks eliminates false sharing in hot fields. Bitmap
// is touched across threads via atomic fetch_or only at the inter-thread border.

#pragma once

#include "propagate.hpp"

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>

namespace cl {

constexpr size_t   CHUNK_CELLS         = 64;
constexpr uint16_t SIGNIFICANT_DELTA   = 4;
constexpr uint8_t  STABLE_THRESH       = 8;
constexpr uint16_t HYSTERESIS_DELTA    = 16;

// W1 full refactor: DirtyTissue_t<T> generic in T (uint8/16/32).
// Backward compat: `DirtyTissue` keeps referring to the uint16_t case via the
// type alias below. Stability and dirty bitmap are uint8_t/uint64_t regardless
// of T (they are meta-structures, not payload).
template <typename T>
struct DirtyTissue_t {
    HotField<T> a;            // value (prev/next via swap)
    HotField<T> b;
    uint8_t*    stability = nullptr;   // counter[n] — saturated at 0xFF
    uint64_t*   dirty     = nullptr;   // bitmap, n_chunks bits
    size_t      n         = 0;
    size_t      n_chunks  = 0;
    size_t      n_words   = 0;

    // W3: runtime-tunable threshold. Default = canonical
    // SIGNIFICANT_DELTA (4). Field type must be comparable to delta = |new - old|
    // (also T). For T=uint16_t keeps uint16_t SIGNIFICANT_DELTA.
    T           sig_delta = (T)SIGNIFICANT_DELTA;
};

using DirtyTissue = DirtyTissue_t<uint16_t>;  // 100% backward compat

// alloc_dirty — allocates tissue + dirty bitmap + stability counters.
// On any allocation failure, frees what was allocated and returns an empty struct.
// Caller MUST check d.a.base != nullptr before use.
//
// W1 full refactor: alloc_dirty_t<T> template version; alloc_dirty is the wrapper
// for T=uint16_t (100% backward compat).
template <typename T>
inline DirtyTissue_t<T> alloc_dirty_t(size_t n) {
    DirtyTissue_t<T> d;
    if (n == 0) return d;
    d.n        = n;
    d.n_chunks = (n + CHUNK_CELLS - 1) / CHUNK_CELLS;
    d.n_words  = (d.n_chunks + 63) / 64;
    d.a = alloc_hf<T>(n, /*halo=*/2);
    if (!d.a.base) { d = {}; return d; }
    d.b = alloc_hf<T>(n, 2);
    if (!d.b.base) { free_hf<T>(d.a); d = {}; return d; }

    size_t stab_bytes = (n + 63) & ~size_t(63);
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
        free_hf<T>(d.a); free_hf<T>(d.b);
#ifdef _WIN32
        if (d.stability) VirtualFree(d.stability, 0, MEM_RELEASE);
        if (d.dirty)     VirtualFree(d.dirty, 0, MEM_RELEASE);
#else
        free(d.stability); free(d.dirty);
#endif
        d = {};
        return d;
    }
    std::memset(d.stability, 0, n);
    std::memset(d.dirty, 0xFF, dirty_bytes);
    return d;
}

template <typename T>
inline void free_dirty_t(DirtyTissue_t<T>& d) {
    free_hf<T>(d.a);
    free_hf<T>(d.b);
#ifdef _WIN32
    if (d.stability) VirtualFree(d.stability, 0, MEM_RELEASE);
    if (d.dirty)     VirtualFree(d.dirty, 0, MEM_RELEASE);
#else
    free(d.stability);
    free(d.dirty);
#endif
    d = {};
}

// Backward-compat wrappers — uint16_t default.
inline DirtyTissue alloc_dirty(size_t n) { return alloc_dirty_t<uint16_t>(n); }
inline void        free_dirty(DirtyTissue& d) { free_dirty_t<uint16_t>(d); }

// === bitmap helpers ===
static inline bool   chunk_is_dirty(const uint64_t* bits, size_t chunk) {
    return (bits[chunk / 64] & (1ULL << (chunk % 64))) != 0;
}
static inline void   chunk_set_dirty(uint64_t* bits, size_t chunk) {
    bits[chunk / 64] |= (1ULL << (chunk % 64));
}
static inline void   chunk_set_dirty_atomic(uint64_t* bits, size_t chunk) {
    auto* w = reinterpret_cast<std::atomic<uint64_t>*>(&bits[chunk / 64]);
    w->fetch_or(1ULL << (chunk % 64), std::memory_order_relaxed);
}
static inline void   chunk_clear_dirty(uint64_t* bits, size_t chunk) {
    bits[chunk / 64] &= ~(1ULL << (chunk % 64));
}

// counts chunks with bit set in [chunk_lo, chunk_hi)
inline size_t count_dirty_chunks(const uint64_t* bits, size_t chunk_lo, size_t chunk_hi) {
    size_t cnt = 0;
    for (size_t c = chunk_lo; c < chunk_hi; ++c) {
        if (chunk_is_dirty(bits, c)) ++cnt;
    }
    return cnt;
}

// checksum for correctness verification
inline uint64_t checksum_hf16_range(const HotField16& h, size_t lo, size_t hi) {
    uint64_t s = 0;
    for (size_t i = lo; i < hi; ++i) s += h.data[i];
    return s;
}

// L1 distance between two HotField16 — measures divergence between
// naive and dirty versions (correctness)
inline uint64_t l1_distance(const HotField16& A, const HotField16& B) {
    uint64_t s = 0;
    size_t n = A.n < B.n ? A.n : B.n;
    for (size_t i = 0; i < n; ++i) {
        int da = (int)A.data[i];
        int db = (int)B.data[i];
        s += (uint64_t)(da > db ? da - db : db - da);
    }
    return s;
}

} // namespace cl
