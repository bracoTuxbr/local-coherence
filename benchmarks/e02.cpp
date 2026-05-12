// e02.cpp — 1D propagation, sweep over tissue size and tile size
//
// proposed gate-zero: ns/cell per generation <= 10 ns in at least one combination
//                    (n_cells, tile) that fits in L2.

#include "runtime.hpp"
#include "propagate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
  #define CL_NOINLINE __declspec(noinline)
#else
  #define CL_NOINLINE __attribute__((noinline))
#endif

using namespace cl;

struct Stats {
    double median, mean, p95, p99, stdev, min, max;
};
static Stats summarize(std::vector<double>& xs) {
    Stats s{};
    if (xs.empty()) return s;
    std::sort(xs.begin(), xs.end());
    s.min = xs.front(); s.max = xs.back();
    s.median = xs[xs.size()/2];
    size_t i95 = std::min(xs.size()-1, (size_t)((double)xs.size() * 0.95));
    size_t i99 = std::min(xs.size()-1, (size_t)((double)xs.size() * 0.99));
    s.p95 = xs[i95]; s.p99 = xs[i99];
    double sum = 0; for (double v : xs) sum += v;
    s.mean = sum / (double)xs.size();
    double sq = 0; for (double v : xs) sq += (v - s.mean) * (v - s.mean);
    s.stdev = std::sqrt(sq / (double)xs.size());
    return s;
}

struct RunResult {
    size_t n_cells;
    size_t tile;
    int    generations;
    Stats  ns_per_cell_per_gen;
    uint64_t checksum;
};

static CL_NOINLINE RunResult bench_one(size_t n_cells, size_t tile,
                                       int generations, int n_runs,
                                       const TscClock& clk) {
    RunResult res{};
    res.n_cells = n_cells;
    res.tile = tile;
    res.generations = generations;

    auto a = alloc_hf16(n_cells);
    auto b = alloc_hf16(n_cells);
    if (!a.data || !b.data) {
        std::fprintf(stderr, "alloc fail\n");
        return res;
    }
    seed_hf16(a, 0xC0FFEEu + (uint32_t)n_cells);
    // clear b
    for (size_t i = 0; i < b.n; ++i) b.data[i] = 0;

    // warmup: 5 runs of N generations
    for (int w = 0; w < 5; ++w) {
        propagate_1d_tiled(a, b, tile);
        swap_hf16(a, b);
    }

    std::vector<double> ns_per_cell_per_gen;
    ns_per_cell_per_gen.reserve(n_runs);
    uint64_t cs = 0;

    for (int r = 0; r < n_runs; ++r) {
        cpu_serialize();
        uint64_t t0 = TscClock::now();
        for (int g = 0; g < generations; ++g) {
            propagate_1d_tiled(a, b, tile);
            swap_hf16(a, b);
        }
        uint64_t t1 = TscClock::now();
        cpu_serialize();
        double ns = clk.to_ns(t1 - t0);
        double per = ns / ((double)n_cells * (double)generations);
        ns_per_cell_per_gen.push_back(per);
        cs ^= checksum_hf16(a);
    }

    res.ns_per_cell_per_gen = summarize(ns_per_cell_per_gen);
    res.checksum = cs;
    free_hf16(a);
    free_hf16(b);
    return res;
}

int main(int argc, char** argv) {
    int    n_runs      = (argc > 1) ? std::atoi(argv[1]) : 20;
    int    generations = (argc > 2) ? std::atoi(argv[2]) : 50;
    int    pin_core    = (argc > 3) ? std::atoi(argv[3]) : 0;

    auto info = detect_cpu();
    std::fprintf(stderr, "=== local-coherence / E02 1D propagation ===\n");
    print_cpu(info);
    std::fprintf(stderr, "n_runs=%d generations=%d pin_core=%d\n",
                 n_runs, generations, pin_core);

    boost_process_priority();
    pin_thread_to_core(pin_core);
    boost_thread_priority();

    TscClock clk; clk.calibrate();
    std::fprintf(stderr, "tsc: ns_per_tick=%.6f invariant=%s\n\n",
                 clk.ns_per_tick, clk.invariant ? "yes" : "no");

    // tissue sizes (in uint16_t):
    //   16K  =  32 KB  (fits in L1 32KB)
    //   64K  = 128 KB  (exceeds L1, in L2)
    //   256K = 512 KB  (in L2)
    //   1M   =   2 MB  (in L3)
    //   4M   =   8 MB  (>L3, goes to DRAM)
    size_t sizes[] = { 16*1024, 64*1024, 256*1024, 1024*1024, 4*1024*1024 };

    // tiles (in uint16_t): 0 = no tiling
    size_t tiles[] = { 0, 4*1024, 16*1024, 64*1024, 256*1024 };

    std::printf("metric,n_cells,bytes_kb,tile,tile_bytes_kb,n_runs,gen,median_ns_cell,p95,p99,stdev,min,max,checksum\n");

    for (size_t n : sizes) {
        for (size_t tile : tiles) {
            // if tile >= n, equivalent to no tile
            if (tile != 0 && tile >= n) continue;
            auto r = bench_one(n, tile, generations, n_runs, clk);
            const auto& s = r.ns_per_cell_per_gen;
            double kb_n    = (double)(n * sizeof(uint16_t)) / 1024.0;
            double kb_tile = (double)(tile * sizeof(uint16_t)) / 1024.0;
            std::printf("propagate_1d,%zu,%.1f,%zu,%.1f,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,0x%llx\n",
                n, kb_n, tile, kb_tile, n_runs, generations,
                s.median, s.p95, s.p99, s.stdev, s.min, s.max,
                (unsigned long long)r.checksum);
            std::fprintf(stderr, "  n=%-8zu tile=%-8zu  median=%6.3f ns/cell  p99=%6.3f  stdev=%.3f\n",
                n, tile, s.median, s.p99, s.stdev);
        }
    }
    return 0;
}
