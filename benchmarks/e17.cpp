// e17.cpp — roofline cache hierarchy via working set sweep
//
// Practical substitute for PMU counters on Windows without perf/VTune/admin.
//
// Thesis: memory-bound workload reveals cache hierarchy via discontinuities
// in ns/cell when the working set crosses the L1/L2/L3/DRAM boundary. For Ryzen 5 7520U:
//   L1d = 32 KiB / core    (data up to ~16K uint16 cells)
//   L2  = 512 KiB / core   (data up to ~256K cells)
//   L3  = 4 MiB shared     (data up to ~2M cells)
//   DRAM = LPDDR5-5500     (beyond)
//
// Procedure:
//   - varies n: 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M, 2M, 4M, 8M, 16M
//   - two uint16 arrays of size n -> effective working set = 2*n*2 bytes
//   - K consecutive propagations measuring median ns/cell
//   - prints CSV: n,ws_kb,ns_cell_median,ns_cell_p95,bandwidth_gbs
//   - effective bandwidth = (2 * n * 2) / median_ns / 1e9 GB/s
//
// Heuristic to identify crossings:
//   - L1->L2: leap > 1.5x when ws crosses ~32KB
//   - L2->L3: leap > 1.3x when ws crosses ~512KB
//   - L3->DRAM: leap > 2x when ws crosses ~4MB

#include "runtime.hpp"
#include "propagate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cl;

struct RoofResult {
    size_t n;
    size_t ws_bytes;
    double ns_cell_median;
    double ns_cell_p95;
    double bandwidth_gbs;
    int    cache_level; // 1=L1, 2=L2, 3=L3, 4=DRAM
};

static double median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static double pct(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    size_t i = std::min(v.size() - 1, (size_t)((double)v.size() * p));
    return v[i];
}

static int infer_level(size_t ws_bytes) {
    if (ws_bytes <= 32 * 1024) return 1;
    if (ws_bytes <= 512 * 1024) return 2;
    if (ws_bytes <= 4 * 1024 * 1024) return 3;
    return 4;
}

static RoofResult run_one(size_t n, int n_runs, int gens_per_run, const TscClock& clk) {
    auto a = alloc_hf16(n, 1);
    auto b = alloc_hf16(n, 1);
    if (!a.base || !b.base) {
        std::fprintf(stderr, "ERROR: alloc failed n=%zu\n", n);
        return {n, 0, 0, 0, 0, 0};
    }
    seed_hf16(a, 0xC0FFEEu);
    // zeroed halo
    for (size_t i = 0; i < a.halo; ++i) {
        a.data[-(ptrdiff_t)(i+1)] = 0;
        a.data[a.n + i] = 0;
        b.data[-(ptrdiff_t)(i+1)] = 0;
        b.data[b.n + i] = 0;
    }

    std::vector<double> ns_cells;
    ns_cells.reserve(n_runs);

    HotField16* prev = &a;
    HotField16* next = &b;

    // warm-up
    for (int g = 0; g < 3; ++g) {
        propagate_1d(*prev, *next);
        std::swap(prev, next);
    }

    for (int run = 0; run < n_runs; ++run) {
        uint64_t t0 = clk.now();
        for (int g = 0; g < gens_per_run; ++g) {
            propagate_1d(*prev, *next);
            std::swap(prev, next);
        }
        uint64_t t1 = clk.now();
        double total_ns = (double)(t1 - t0) * clk.ns_per_tick;
        double ns_cell = total_ns / ((double)n * (double)gens_per_run);
        ns_cells.push_back(ns_cell);
    }

    RoofResult r;
    r.n = n;
    r.ws_bytes = 2 * n * sizeof(uint16_t);  // 2 buffers
    r.ns_cell_median = median(ns_cells);
    r.ns_cell_p95 = pct(ns_cells, 0.95);
    // effective bandwidth: each gen reads n*2 bytes from prev + writes n*2 to next
    // = 4 bytes/cell. ns_cell -> bytes/sec = 4 / (ns_cell * 1e-9) = 4e9 / ns_cell
    r.bandwidth_gbs = 4.0 / r.ns_cell_median;
    r.cache_level = infer_level(r.ws_bytes);

    free_hf16(a); free_hf16(b);
    return r;
}

int main(int argc, char** argv) {
    int n_runs = (argc > 1) ? std::atoi(argv[1]) : 7;
    int gens_per_run = (argc > 2) ? std::atoi(argv[2]) : 100;

    auto info = detect_cpu();
    std::fprintf(stderr, "=== local-coherence / E17 roofline cache hierarchy ===\n");
    print_cpu(info);
    std::fprintf(stderr, "n_runs=%d gens_per_run=%d\n\n", n_runs, gens_per_run);

    boost_process_priority();
    pin_thread_to_core(0);
    boost_thread_priority();

    TscClock clk; clk.calibrate();

    // size sweep: 4K -> 16M (powers of 2)
    std::vector<size_t> sizes;
    for (size_t n = 4096; n <= (1u << 24); n *= 2) sizes.push_back(n);

    std::printf("n,ws_bytes,ws_kb,ns_cell_median,ns_cell_p95,bandwidth_gbs,cache_level\n");
    std::fprintf(stderr,
        "  n         ws_kb     ns/cell  p95     GB/s    level\n");

    std::vector<RoofResult> results;
    for (size_t n : sizes) {
        // adjust gens so each run lasts at least ~50ms (stabilize TSC)
        int gens = gens_per_run;
        if (n < 65536) gens = std::max(gens_per_run, (int)(50.0 * 1e6 * 0.5 / (double)n));
        if (gens > 5000) gens = 5000;

        auto r = run_one(n, n_runs, gens, clk);
        results.push_back(r);

        const char* level_str =
            r.cache_level == 1 ? "L1" :
            r.cache_level == 2 ? "L2" :
            r.cache_level == 3 ? "L3" : "DRAM";
        std::fprintf(stderr,
            "  %-9zu %-9.1f %.3f    %.3f   %.2f    %s\n",
            r.n, r.ws_bytes / 1024.0,
            r.ns_cell_median, r.ns_cell_p95, r.bandwidth_gbs,
            level_str);
        std::printf("%zu,%zu,%.1f,%.6f,%.6f,%.3f,%s\n",
            r.n, r.ws_bytes, r.ws_bytes / 1024.0,
            r.ns_cell_median, r.ns_cell_p95, r.bandwidth_gbs, level_str);
    }

    // crossings analysis: detects where ns/cell leaps
    std::fprintf(stderr, "\n--- crossings detected ---\n");
    for (size_t i = 1; i < results.size(); ++i) {
        double ratio = results[i].ns_cell_median / results[i-1].ns_cell_median;
        if (ratio >= 1.3) {
            std::fprintf(stderr, "  ws=%.0f KB -> %.0f KB: ns/cell %.3f -> %.3f (%.2fx)\n",
                results[i-1].ws_bytes / 1024.0,
                results[i].ws_bytes / 1024.0,
                results[i-1].ns_cell_median,
                results[i].ns_cell_median,
                ratio);
        }
    }

    // print peak bandwidth (L1 regime)
    double peak_gbs = 0;
    double dram_gbs = 0;
    for (auto& r : results) {
        if (r.cache_level == 1 && r.bandwidth_gbs > peak_gbs) peak_gbs = r.bandwidth_gbs;
        if (r.cache_level == 4 && r.bandwidth_gbs > dram_gbs) dram_gbs = r.bandwidth_gbs;
    }
    std::fprintf(stderr, "\n--- bandwidth summary ---\n");
    std::fprintf(stderr, "  peak (L1) = %.2f GB/s\n", peak_gbs);
    std::fprintf(stderr, "  DRAM      = %.2f GB/s\n", dram_gbs);
    std::fprintf(stderr, "  ratio     = %.1fx\n", peak_gbs / std::max(dram_gbs, 0.01));

    return 0;
}
