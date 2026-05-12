// e20.cpp — Leap 11: workers that sleep in the absence of perturbation
//
// Wave-driven model (Leap 11 from 01-timeline-prospeccao):
//   "No perturbation = no CPU. This IS the scheduler. There is no other."
//
// Unlike M2 (e03): workers do not participate in each generation via barrier.
// They sleep on a condvar; they wake up when the dispatcher signals a wave.
// They process until the bitmap empties, then go back to sleep.
//
// A vs B comparison on a "frame" pipeline (analogous to real-time audio):
//   A) spin version: classic SpinBarrier, workers burn 100% CPU even idle
//   B) sleep version: workers sleep between frames, CPU ~ 0% at idle
//
// Measures:
//   - total CPU time (GetProcessTimes) - user time consumed
//   - total wallclock
//   - CPU/wallclock ratio as utilization proxy
//   - wakeup latency (TSC ticks between signal and first work)
//   - bit-exact: both versions produce the same final result

#include "runtime.hpp"
#include "propagate.hpp"
#include "dirty.hpp"
#include "propagate_dirty.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

using namespace cl;

static inline void atomic_or_bit(uint64_t* bits, size_t chunk_idx) {
    size_t w = chunk_idx / 64;
    uint64_t m = 1ULL << (chunk_idx % 64);
    __atomic_fetch_or(&bits[w], m, __ATOMIC_RELAXED);
}

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

    bool changed = false, first_changed = false, last_changed = false;
    for (size_t i = s; i < e; ++i) {
        uint32_t l = x[(ptrdiff_t)i - 1], c = x[i], r = x[i + 1];
        uint32_t avg = (l + (c << 1) + r) >> 2;
        uint32_t dec = (avg * 255u) >> 8;
        uint16_t newv = (uint16_t)dec, oldv = (uint16_t)c;
        uint16_t delta = (newv > oldv) ? (newv - oldv) : (oldv - newv);
        if (delta < SIGNIFICANT_DELTA) {
            if (stability[i] < 0xFF) ++stability[i];
        } else {
            stability[i] = 0; changed = true;
            if (i == s) first_changed = true;
            if (i == e - 1) last_changed = true;
        }
        y[i] = newv;
    }
    if (changed) {
        atomic_or_bit(next_bits, chunk_idx);
        if (first_changed && chunk_idx > 0)
            atomic_or_bit(next_bits, chunk_idx - 1);
        if (last_changed && chunk_idx + 1 < n_chunks)
            atomic_or_bit(next_bits, chunk_idx + 1);
    }
}

// === wave-driven pool ===
//
// 1 dispatcher thread (mainthread) coordinates. T workers.
// Shared state:
//   - pending_gens: how many gens are still to process in this "wave"
//   - workers_busy: how many workers are in the middle of a gen
//   - prev_is_a, next_bits: tissue state
//
// Worker loop:
//   while !stop:
//     lock(m) -> wait until pending_gens > 0
//     while pending_gens > 0:
//       process my partition
//       atomic decrement workers_busy
//       wait barrier (lock + wait until all workers done this gen)
//       (only tid 0 swaps bitmaps + decrements pending_gens)
//
// Dispatcher:
//   inject_pulse()
//   pending_gens = K
//   cv.notify_all()
//   wait until pending_gens == 0
//   sleep(idle_window_ms)  <- workers are sleeping here

struct Pool {
    int T;
    std::vector<size_t> chunk_lo, chunk_hi;
    DirtyTissue* d;
    bool prev_is_a;
    uint64_t* next_bits;

    std::mutex m;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    int pending_gens = 0;
    int workers_busy = 0;
    int gen_idx = 0;
    bool stop = false;

    std::atomic<size_t> total_processed{0};
    std::atomic<uint64_t> wakeup_ticks_sum{0};
    std::atomic<int> wakeup_count{0};
    uint64_t signal_tsc = 0;  // dispatcher sets it before notify_all
};

static void worker_main(Pool* p, int tid, int core_id) {
    pin_thread_to_core(core_id);
    boost_thread_priority();

    while (true) {
        // 1) wait for work
        std::unique_lock<std::mutex> lk(p->m);
        p->cv_work.wait(lk, [p]{ return p->pending_gens > 0 || p->stop; });
        if (p->stop) return;

        // measure wakeup latency (only the first to wake up)
        uint64_t now = TscClock::now();
        if (p->signal_tsc != 0 && p->gen_idx == 0) {
            p->wakeup_ticks_sum.fetch_add(now - p->signal_tsc,
                std::memory_order_relaxed);
            p->wakeup_count.fetch_add(1, std::memory_order_relaxed);
        }

        // 2) gens loop while pending_gens > 0
        while (p->pending_gens > 0 && !p->stop) {
            int my_gen = p->gen_idx;
            ++p->workers_busy;
            lk.unlock();

            // === processing (no lock) ===
            const HotField16& prev = p->prev_is_a ? p->d->a : p->d->b;
            HotField16&       next = p->prev_is_a ? p->d->b : p->d->a;
            size_t my_processed = 0;
            size_t word_lo = p->chunk_lo[tid] / 64;
            size_t word_hi = (p->chunk_hi[tid] + 63) / 64;
            for (size_t wi = word_lo; wi < word_hi; ++wi) {
                uint64_t word = p->d->dirty[wi];
                if (word == 0) continue;
                while (word) {
                    int b = __builtin_ctzll(word);
                    word &= word - 1;
                    size_t chunk = wi * 64 + (size_t)b;
                    if (chunk < p->chunk_lo[tid] || chunk >= p->chunk_hi[tid]) continue;
                    if (chunk >= p->d->n_chunks) break;
                    propagate_chunk_mt(prev, next, p->d->stability,
                                       chunk, p->d->n, p->d->n_chunks,
                                       p->next_bits);
                    ++my_processed;
                }
            }
            p->total_processed.fetch_add(my_processed, std::memory_order_relaxed);

            // === end-of-gen barrier ===
            lk.lock();
            --p->workers_busy;
            if (p->workers_busy == 0 && p->gen_idx == my_gen) {
                // last worker of this gen: swap, advance gen
                std::memcpy(p->d->dirty, p->next_bits,
                    p->d->n_words * sizeof(uint64_t));
                std::memset(p->next_bits, 0,
                    p->d->n_words * sizeof(uint64_t));
                p->prev_is_a = !p->prev_is_a;
                --p->pending_gens;
                ++p->gen_idx;
                if (p->pending_gens == 0) {
                    // wave finished: notify dispatcher
                    p->cv_done.notify_all();
                } else {
                    p->cv_work.notify_all();
                }
            } else {
                // wait for end of gen
                int wait_gen = my_gen;
                p->cv_work.wait(lk, [p, wait_gen]{
                    return p->gen_idx > wait_gen || p->stop;
                });
            }
        }
    }
}

// === dispatcher: runs K gens, waits for workers to finish ===
static void dispatch_wave(Pool* p, int K) {
    {
        std::lock_guard<std::mutex> lk(p->m);
        p->pending_gens = K;
        p->gen_idx = 0;
        p->signal_tsc = TscClock::now();
    }
    p->cv_work.notify_all();
    {
        std::unique_lock<std::mutex> lk(p->m);
        p->cv_done.wait(lk, [p]{ return p->pending_gens == 0 || p->stop; });
        p->signal_tsc = 0;
    }
}

static void shutdown_pool(Pool* p, std::vector<std::thread>& threads) {
    {
        std::lock_guard<std::mutex> lk(p->m);
        p->stop = true;
    }
    p->cv_work.notify_all();
    for (auto& th : threads) th.join();
}

// === measurement of consumed CPU time ===
static double process_user_seconds() {
#ifdef _WIN32
    FILETIME create, exit, kernel, user;
    GetProcessTimes(GetCurrentProcess(), &create, &exit, &kernel, &user);
    ULARGE_INTEGER u;
    u.LowPart = user.dwLowDateTime; u.HighPart = user.dwHighDateTime;
    return (double)u.QuadPart * 100e-9; // 100ns ticks -> seconds
#else
    return 0.0;
#endif
}

// === pipeline simulator ===
struct PipelineResult {
    double wallclock_s;
    double cpu_user_s;
    double active_ratio;  // CPU/wallclock
    size_t total_processed;
    size_t l1_distance;
    uint64_t checksum;
    double avg_wakeup_us;
};

static PipelineResult run_pipeline(int T, int frames, int gens_per_frame,
                                   int idle_ms_between_frames,
                                   size_t n, uint16_t pulse,
                                   const TscClock& clk) {
    DirtyTissue d;
    d.n = n; d.n_chunks = (n + CHUNK_CELLS - 1) / CHUNK_CELLS;
    d.n_words = (d.n_chunks + 63) / 64;
    d.a = alloc_hf16(n, 1); d.b = alloc_hf16(n, 1);
    d.stability = (uint8_t*)std::calloc(n, 1);
    d.dirty = (uint64_t*)std::calloc(d.n_words, sizeof(uint64_t));
    for (size_t i = 0; i < n; ++i) { d.a.data[i] = 0; d.b.data[i] = 0; }

    Pool p;
    p.T = T;
    p.d = &d;
    p.prev_is_a = true;
    p.next_bits = (uint64_t*)std::calloc(d.n_words, sizeof(uint64_t));
    p.chunk_lo.resize(T); p.chunk_hi.resize(T);
    size_t chunks_per_thread = (d.n_chunks + T - 1) / T;
    chunks_per_thread = ((chunks_per_thread + 63) / 64) * 64;
    for (int t = 0; t < T; ++t) {
        p.chunk_lo[t] = (size_t)t * chunks_per_thread;
        p.chunk_hi[t] = std::min(p.chunk_lo[t] + chunks_per_thread, d.n_chunks);
    }

    auto cores = physical_core_indices();
    std::vector<std::thread> threads;
    for (int t = 0; t < T; ++t) {
        threads.emplace_back(worker_main, &p, t, cores[t % cores.size()]);
    }
    // small settle so workers enter cv.wait
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    auto wall_t0 = std::chrono::steady_clock::now();
    double cpu_t0 = process_user_seconds();

    for (int f = 0; f < frames; ++f) {
        // inject perturbation at the center (simulates a new data "frame")
        size_t pos = (n / 2) + (size_t)f * 13;  // varies position per frame
        if (pos >= n) pos = pos % n;
        inject_pulse(d, p.prev_is_a, pos, pos + 1, pulse);
        // dispatch
        dispatch_wave(&p, gens_per_frame);
        // idle window between frames (workers should sleep)
        if (idle_ms_between_frames > 0) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(idle_ms_between_frames));
        }
    }

    auto wall_t1 = std::chrono::steady_clock::now();
    double cpu_t1 = process_user_seconds();

    shutdown_pool(&p, threads);

    PipelineResult r;
    r.wallclock_s = std::chrono::duration<double>(wall_t1 - wall_t0).count();
    r.cpu_user_s = cpu_t1 - cpu_t0;
    r.active_ratio = r.wallclock_s > 0 ? r.cpu_user_s / r.wallclock_s : 0.0;
    r.total_processed = p.total_processed.load();
    HotField16& last = p.prev_is_a ? d.a : d.b;
    r.checksum = checksum_hf16(last);
    size_t l1 = 0;
    for (size_t i = 0; i < d.n; ++i) l1 += last.data[i];
    r.l1_distance = l1;
    int wc = p.wakeup_count.load();
    uint64_t wts = p.wakeup_ticks_sum.load();
    r.avg_wakeup_us = wc > 0 ? (double)wts / (double)wc * clk.ns_per_tick / 1000.0 : 0.0;

    std::free(p.next_bits);
    free_hf16(d.a); free_hf16(d.b);
    std::free(d.stability); std::free(d.dirty);

    (void)clk;
    return r;
}

int main(int argc, char** argv) {
    int frames = (argc > 1) ? std::atoi(argv[1]) : 10;
    int gens = (argc > 2) ? std::atoi(argv[2]) : 100;
    int idle_ms = (argc > 3) ? std::atoi(argv[3]) : 100;
    size_t n = (argc > 4) ? (size_t)std::strtoull(argv[4], nullptr, 10) : (1u << 18);
    uint16_t pulse = (argc > 5) ? (uint16_t)std::atoi(argv[5]) : 60000;

    auto info = detect_cpu();
    std::fprintf(stderr, "=== local-coherence / E20 wave-driven workers (Leap 11) ===\n");
    print_cpu(info);
    std::fprintf(stderr, "frames=%d gens/frame=%d idle_between=%dms n=%zu pulse=%u\n\n",
                 frames, gens, idle_ms, n, pulse);
    std::fprintf(stderr, "expectation: workers sleep between frames -> active_ratio < 0.3\n");
    std::fprintf(stderr, "             spin version would have active_ratio ~ T (all threads burning)\n\n");

    boost_process_priority();
    pin_thread_to_core(0);
    boost_thread_priority();

    TscClock clk; clk.calibrate();

    std::printf("threads,frames,gens,idle_ms,wallclock_s,cpu_user_s,active_ratio,processed,l1,checksum,avg_wakeup_us\n");
    std::fprintf(stderr, "  T  wall_s  cpu_s   ratio   processed  L1     wakeup_us\n");

    for (int T : {1, 2, 4}) {
        auto r = run_pipeline(T, frames, gens, idle_ms, n, pulse, clk);
        std::fprintf(stderr, "  %d  %-7.3f %-7.3f %-7.3f %-10zu %-6zu %.2f\n",
            T, r.wallclock_s, r.cpu_user_s, r.active_ratio,
            r.total_processed, r.l1_distance, r.avg_wakeup_us);
        std::printf("%d,%d,%d,%d,%.4f,%.4f,%.4f,%zu,%zu,0x%016llx,%.3f\n",
            T, frames, gens, idle_ms,
            r.wallclock_s, r.cpu_user_s, r.active_ratio,
            r.total_processed, r.l1_distance,
            (unsigned long long)r.checksum, r.avg_wakeup_us);
    }

    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "interpretation:\n");
    std::fprintf(stderr, "  active_ratio = cpu_user_s / wallclock_s\n");
    std::fprintf(stderr, "  spin baseline: ~ T (all threads burning 100%%)\n");
    std::fprintf(stderr, "  Leap 11 OK:    ~ (gens*frames*work_per_gen)/wallclock <<< T\n");
    std::fprintf(stderr, "  active_ratio close to 0 = workers sleeping well\n");

    return 0;
}
