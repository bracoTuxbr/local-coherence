// e19.cpp — multi-thread + dirty + freeze combined (M2 x M4)
//
// Combination not explicitly tested before: each thread owns a chunk range,
// sweeps only its part of the dirty bitmap, processes dirty chunks, propagates
// cross-thread dirty via atomic OR.
//
// Matches the 3 axes of the final Leap from 01-timeline-prospeccao:
//   - tissue (HotField16 + dirty bitmap)
//   - waves (propagates only where it changed)
//   - physical cores (T pinned threads, chunk-aligned partitioning)
//
// Output: effective ns/cell/gen (only cells in dirty chunks), speedup vs T=1,
// active_pct over time, bit-exact match vs single-thread reference.

#include "runtime.hpp"
#include "propagate.hpp"
#include "dirty.hpp"
#include "propagate_dirty.hpp"
#include "barrier.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using namespace cl;

// === atomic OR on a bitmap word ===
static inline void atomic_or_bit(uint64_t* bits, size_t chunk_idx) {
    size_t w = chunk_idx / 64;
    uint64_t m = 1ULL << (chunk_idx % 64);
    __atomic_fetch_or(&bits[w], m, __ATOMIC_RELAXED);
}

// === single-chunk processing in MT mode ===
// Identical to single-thread propagate_chunk, but cross-thread dirty propagation
// uses atomic_or on the next bitmap. Does not write to the current bitmap.
__attribute__((noinline))
static void propagate_chunk_mt(const HotField16& prev, HotField16& next,
                               uint8_t* stability,
                               size_t chunk_idx, size_t n,
                               size_t n_chunks,
                               uint64_t* next_bits) {
    size_t s = chunk_idx * CHUNK_CELLS;
    size_t e = std::min(s + CHUNK_CELLS, n);
    const uint16_t* __restrict__ x = prev.data;
    uint16_t* __restrict__ y = next.data;

    bool changed = false;
    bool first_changed = false;
    bool last_changed = false;

    for (size_t i = s; i < e; ++i) {
        uint32_t l = x[(ptrdiff_t)i - 1];
        uint32_t c = x[i];
        uint32_t r = x[i + 1];
        uint32_t avg = (l + (c << 1) + r) >> 2;
        uint32_t dec = (avg * 255u) >> 8;
        uint16_t newv = (uint16_t)dec;
        uint16_t oldv = (uint16_t)c;
        uint16_t delta = (newv > oldv) ? (newv - oldv) : (oldv - newv);

        if (delta < SIGNIFICANT_DELTA) {
            if (stability[i] < 0xFF) ++stability[i];
        } else {
            stability[i] = 0;
            changed = true;
            if (i == s)         first_changed = true;
            if (i == e - 1)     last_changed = true;
        }
        y[i] = newv;
    }

    if (changed) {
        atomic_or_bit(next_bits, chunk_idx);
        if (first_changed && chunk_idx > 0) {
            atomic_or_bit(next_bits, chunk_idx - 1);
        }
        if (last_changed && chunk_idx + 1 < n_chunks) {
            atomic_or_bit(next_bits, chunk_idx + 1);
        }
    }
}

// === worker ===
struct Worker {
    int tid;
    int core_id;
    size_t chunk_lo;       // first chunk of this thread (inclusive)
    size_t chunk_hi;       // last chunk (exclusive)
    DirtyTissue* d;
    int gens;
    bool* prev_is_a_ptr;   // alternates between gens
    uint64_t** next_bits_ptr; // output bitmap for this gen
    SpinBarrier* barrier;
    std::atomic<bool>* go;
    std::atomic<size_t>* total_processed;
    uint64_t local_t0_ticks;
    uint64_t local_t1_ticks;
};

static void worker_loop(Worker* w) {
    pin_thread_to_core(w->core_id);
    boost_thread_priority();
    while (!w->go->load(std::memory_order_acquire)) _mm_pause();

    uint32_t sense = 0;
    w->local_t0_ticks = TscClock::now();
    size_t my_processed = 0;

    for (int g = 0; g < w->gens; ++g) {
        bool prev_is_a = *(w->prev_is_a_ptr);
        const HotField16& prev = prev_is_a ? w->d->a : w->d->b;
        HotField16&       next = prev_is_a ? w->d->b : w->d->a;
        uint64_t* next_bits = *(w->next_bits_ptr);

        // sweep the chunks owned by this worker
        size_t word_lo = w->chunk_lo / 64;
        size_t word_hi = (w->chunk_hi + 63) / 64;
        for (size_t wi = word_lo; wi < word_hi; ++wi) {
            uint64_t word = w->d->dirty[wi];
            if (word == 0) continue;
            while (word) {
                int b = __builtin_ctzll(word);
                word &= word - 1;
                size_t chunk = wi * 64 + (size_t)b;
                if (chunk < w->chunk_lo || chunk >= w->chunk_hi) continue;
                if (chunk >= w->d->n_chunks) break;
                propagate_chunk_mt(prev, next, w->d->stability,
                                   chunk, w->d->n, w->d->n_chunks,
                                   next_bits);
                ++my_processed;
            }
        }
        // synchronize end of gen
        w->barrier->wait(sense);
        // thread 0 swaps bitmaps + flips prev_is_a (only one does this)
        // the rest wait on this second barrier
        // simplification: tid==0 does it; others wait
        if (w->tid == 0) {
            std::memcpy(w->d->dirty, next_bits, w->d->n_words * sizeof(uint64_t));
            std::memset(next_bits, 0, w->d->n_words * sizeof(uint64_t));
            *(w->prev_is_a_ptr) = !prev_is_a;
        }
        w->barrier->wait(sense);
    }

    w->local_t1_ticks = TscClock::now();
    w->total_processed->fetch_add(my_processed, std::memory_order_relaxed);
}

// === single-thread reference (for bit-exact validation) ===
struct RefRun {
    size_t l1_distance;
    uint64_t checksum;
    double ns_total;
    size_t total_processed;
};

static RefRun run_single_threaded(DirtyTissue& d, int gens, const TscClock& clk) {
    auto next_bits = (uint64_t*)std::calloc(d.n_words, sizeof(uint64_t));
    bool prev_is_a = true;
    uint64_t t0 = clk.now();
    size_t processed = 0;
    for (int g = 0; g < gens; ++g) {
        processed += propagate_dirty_pass(d, prev_is_a, next_bits);
        apply_next_dirty(d, next_bits);
        prev_is_a = !prev_is_a;
    }
    uint64_t t1 = clk.now();
    HotField16& last = prev_is_a ? d.a : d.b;
    uint64_t cs = checksum_hf16(last);
    // L1 distance vs zero (same metric as e05)
    size_t l1 = 0;
    for (size_t i = 0; i < d.n; ++i) l1 += last.data[i];
    std::free(next_bits);
    return {l1, cs, (double)(t1 - t0) * clk.ns_per_tick, processed};
}

// === main ===
int main(int argc, char** argv) {
    size_t n = (argc > 1) ? (size_t)std::strtoull(argv[1], nullptr, 10) : (1u << 20);
    int gens = (argc > 2) ? std::atoi(argv[2]) : 4000;
    uint16_t pulse = (argc > 3) ? (uint16_t)std::atoi(argv[3]) : 60000;

    auto info = detect_cpu();
    std::fprintf(stderr, "=== local-coherence / E19 multi-thread + dirty + freeze ===\n");
    print_cpu(info);
    std::fprintf(stderr, "n=%zu gens=%d pulse=%u\n\n", n, gens, pulse);

    boost_process_priority();
    pin_thread_to_core(0);
    boost_thread_priority();

    TscClock clk; clk.calibrate();

    auto cores = physical_core_indices();
    int max_threads = (int)cores.size();
    if (max_threads > 4) max_threads = 4;
    std::fprintf(stderr, "physical cores: ");
    for (int c : cores) std::fprintf(stderr, "%d ", c);
    std::fprintf(stderr, " (using up to %d)\n\n", max_threads);

    // === reference single-thread ===
    std::fprintf(stderr, "--- reference single-thread (T=1) ---\n");
    DirtyTissue d_ref; d_ref.n = n; d_ref.n_chunks = (n + CHUNK_CELLS - 1) / CHUNK_CELLS;
    d_ref.n_words = (d_ref.n_chunks + 63) / 64;
    d_ref.a = alloc_hf16(n, 1); d_ref.b = alloc_hf16(n, 1);
    d_ref.stability = (uint8_t*)std::calloc(n, 1);
    d_ref.dirty = (uint64_t*)std::calloc(d_ref.n_words, sizeof(uint64_t));
    if (!d_ref.a.base || !d_ref.b.base || !d_ref.stability || !d_ref.dirty) {
        std::fprintf(stderr, "FAIL alloc reference\n"); return 1;
    }
    for (size_t i = 0; i < n; ++i) { d_ref.a.data[i] = 0; d_ref.b.data[i] = 0; }
    inject_pulse(d_ref, true, n/2, n/2 + 1, pulse);
    auto ref = run_single_threaded(d_ref, gens, clk);
    double ref_ns_per_proc = ref.total_processed ? ref.ns_total / (double)ref.total_processed / (double)CHUNK_CELLS : 0.0;
    std::fprintf(stderr, "  total_ms=%.2f processed_chunks=%zu ns/cell_eff=%.3f L1=%zu checksum=0x%016llx\n",
                 ref.ns_total / 1e6, ref.total_processed, ref_ns_per_proc, ref.l1_distance,
                 (unsigned long long)ref.checksum);
    free_hf16(d_ref.a); free_hf16(d_ref.b);
    std::free(d_ref.stability); std::free(d_ref.dirty);

    // === multi-thread runs ===
    std::printf("threads,total_ms,processed_chunks,ns_cell_eff,speedup,l1_distance,checksum,bit_exact\n");
    std::fprintf(stderr, "\n--- multi-thread runs ---\n");
    std::fprintf(stderr, "  T  total_ms  processed   ns/cell_eff   speedup   bit_exact\n");

    for (int T = 1; T <= max_threads; ++T) {
        DirtyTissue d; d.n = n; d.n_chunks = (n + CHUNK_CELLS - 1) / CHUNK_CELLS;
        d.n_words = (d.n_chunks + 63) / 64;
        d.a = alloc_hf16(n, 1); d.b = alloc_hf16(n, 1);
        d.stability = (uint8_t*)std::calloc(n, 1);
        d.dirty = (uint64_t*)std::calloc(d.n_words, sizeof(uint64_t));
        for (size_t i = 0; i < n; ++i) { d.a.data[i] = 0; d.b.data[i] = 0; }
        inject_pulse(d, true, n/2, n/2 + 1, pulse);

        // partitioning: chunks per thread (aligned to 64-bit words)
        size_t chunks_per_thread = (d.n_chunks + T - 1) / T;
        // round up to a multiple of 64 (1 word) to align the sweep
        chunks_per_thread = ((chunks_per_thread + 63) / 64) * 64;

        SpinBarrier barrier; barrier.init(T);
        std::atomic<bool> go(false);
        std::atomic<size_t> total_processed(0);
        bool prev_is_a = true;
        auto next_bits_buf = (uint64_t*)std::calloc(d.n_words, sizeof(uint64_t));
        uint64_t* next_bits = next_bits_buf;

        std::vector<Worker> workers(T);
        std::vector<std::thread> threads;
        for (int t = 0; t < T; ++t) {
            workers[t].tid = t;
            workers[t].core_id = cores[t % cores.size()];
            workers[t].chunk_lo = (size_t)t * chunks_per_thread;
            workers[t].chunk_hi = std::min(workers[t].chunk_lo + chunks_per_thread, d.n_chunks);
            workers[t].d = &d;
            workers[t].gens = gens;
            workers[t].prev_is_a_ptr = &prev_is_a;
            workers[t].next_bits_ptr = &next_bits;
            workers[t].barrier = &barrier;
            workers[t].go = &go;
            workers[t].total_processed = &total_processed;
        }
        for (int t = 0; t < T; ++t) threads.emplace_back(worker_loop, &workers[t]);
        // small wait for all threads to reach the spin
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        go.store(true, std::memory_order_release);
        for (auto& th : threads) th.join();

        // measure time via TSC from the fastest worker vs the slowest
        uint64_t t_start = workers[0].local_t0_ticks;
        uint64_t t_end = workers[0].local_t1_ticks;
        for (auto& w : workers) {
            if (w.local_t0_ticks < t_start) t_start = w.local_t0_ticks;
            if (w.local_t1_ticks > t_end)   t_end   = w.local_t1_ticks;
        }
        double total_ns = (double)(t_end - t_start) * clk.ns_per_tick;
        size_t processed = total_processed.load();
        double ns_per_cell_eff = processed
            ? total_ns / (double)processed / (double)CHUNK_CELLS : 0.0;
        double speedup = ref.ns_total / std::max(total_ns, 1.0);

        // bit-exact validation
        HotField16& last = prev_is_a ? d.a : d.b;
        uint64_t cs = checksum_hf16(last);
        size_t l1 = 0;
        for (size_t i = 0; i < d.n; ++i) l1 += last.data[i];
        bool bit_exact = (cs == ref.checksum) && (l1 == ref.l1_distance);

        std::fprintf(stderr, "  %d  %-9.2f %-10zu %-13.3f %-9.2fx  %s\n",
                     T, total_ns / 1e6, processed, ns_per_cell_eff, speedup,
                     bit_exact ? "OK" : "FAIL");
        std::printf("%d,%.2f,%zu,%.6f,%.3f,%zu,0x%016llx,%s\n",
                    T, total_ns / 1e6, processed, ns_per_cell_eff, speedup,
                    l1, (unsigned long long)cs, bit_exact ? "OK" : "FAIL");

        std::free(next_bits_buf);
        free_hf16(d.a); free_hf16(d.b);
        std::free(d.stability); std::free(d.dirty);
    }

    return 0;
}
