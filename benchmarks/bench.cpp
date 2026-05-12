// bench.cpp — E01: pure linear scan over a cell tissue
// goal: measure per-cell streaming cost, expose real memory bandwidth
// emits CSV on stdout (header + rows) for parse_uprof.py / spreadsheet
//
// usage:
//   bench.exe [n_cells] [n_runs] [pin_core]
//   default: n_cells=1<<20 (1M cells = 64 MB), n_runs=30, pin_core=0

#include "tissue.hpp"
#include "runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#if defined(_MSC_VER) && !defined(__clang__)
  #define CL_NOINLINE __declspec(noinline)
  #define CL_RESTRICT __restrict
#else
  #define CL_NOINLINE __attribute__((noinline))
  #define CL_RESTRICT __restrict__
#endif

using namespace cl;

// kernel E01: pure linear scan.
// volatile-ish to prevent aggressive optimization. final accumulator used as sink.
static CL_NOINLINE uint64_t scan_linear(Cell* CL_RESTRICT tissue, size_t n) {
    uint64_t acc = 0;
    for (size_t i = 0; i < n; ++i) {
        const Cell& c = tissue[i];
        // touches every field in the cell. sum prevents dead-code elimination.
        acc += (uint64_t)c.acoustic_type
             + (uint64_t)c.confidence
             + (uint64_t)c.energy
             + (uint64_t)c.stability
             + (uint64_t)c.flags
             + (uint64_t)c.timestamp_delta
             + (uint64_t)c.hypothesis_a
             + (uint64_t)c.hypothesis_b
             + (uint64_t)c.score_a
             + (uint64_t)c.score_b;
    }
    return acc;
}

// kernel E01b: pure streaming write (memset-like with our layout)
static CL_NOINLINE void scan_write(Cell* CL_RESTRICT tissue, size_t n, uint16_t mark) {
    for (size_t i = 0; i < n; ++i) {
        tissue[i].stability = mark;
    }
}

struct Stats {
    double median, mean, p95, p99, stdev, min, max;
};

static Stats summarize(std::vector<double>& xs) {
    Stats s{};
    if (xs.empty()) return s;
    std::sort(xs.begin(), xs.end());
    s.min = xs.front();
    s.max = xs.back();
    s.median = xs[xs.size()/2];
    size_t i95 = (size_t)((double)xs.size() * 0.95);
    size_t i99 = (size_t)((double)xs.size() * 0.99);
    if (i95 >= xs.size()) i95 = xs.size()-1;
    if (i99 >= xs.size()) i99 = xs.size()-1;
    s.p95 = xs[i95];
    s.p99 = xs[i99];
    double sum = 0;
    for (double v : xs) sum += v;
    s.mean = sum / (double)xs.size();
    double sq = 0;
    for (double v : xs) sq += (v - s.mean) * (v - s.mean);
    s.stdev = std::sqrt(sq / (double)xs.size());
    return s;
}

int main(int argc, char** argv) {
    size_t n_cells = (argc > 1) ? (size_t)std::strtoull(argv[1], nullptr, 10) : (1ull << 20);
    int    n_runs  = (argc > 2) ? std::atoi(argv[2]) : 30;
    int    pin_core = (argc > 3) ? std::atoi(argv[3]) : 0;
    int    n_warmup = 5;

    auto info = detect_cpu();
    std::fprintf(stderr, "=== local-coherence / bench E01 ===\n");
    print_cpu(info);
    std::fprintf(stderr, "n_cells=%zu (%.2f MB)  n_runs=%d  pin_core=%d\n",
        n_cells, (double)(n_cells * sizeof(Cell)) / (1024.0*1024.0), n_runs, pin_core);

    boost_process_priority();
    if (!pin_thread_to_core(pin_core)) {
        std::fprintf(stderr, "WARN: failed to pin thread to core %d\n", pin_core);
    }
    boost_thread_priority();

    TscClock clk; clk.calibrate();
    std::fprintf(stderr, "tsc: ns_per_tick=%.6f invariant=%s\n",
        clk.ns_per_tick, clk.invariant ? "yes" : "no");

    auto a = alloc_tissue(n_cells);
    if (!a.data) { std::fprintf(stderr, "alloc failed\n"); return 1; }
    std::fprintf(stderr, "alloc: %.2f MB  large_pages=%s\n",
        (double)a.bytes/(1024.0*1024.0), a.used_lpages ? "yes" : "no");

    seed_pattern(a);
    touch_all(a);

    // warmup
    volatile uint64_t sink = 0;
    for (int w = 0; w < n_warmup; ++w) sink ^= scan_linear(a.data, a.n_cells);

    // measurements — READ mode
    std::vector<double> read_ns_per_cell;  read_ns_per_cell.reserve(n_runs);
    std::vector<double> read_gbps;         read_gbps.reserve(n_runs);
    for (int r = 0; r < n_runs; ++r) {
        cpu_serialize();
        uint64_t t0 = TscClock::now();
        sink ^= scan_linear(a.data, a.n_cells);
        uint64_t t1 = TscClock::now();
        cpu_serialize();
        double ns_total = clk.to_ns(t1 - t0);
        double ns_cell  = ns_total / (double)a.n_cells;
        double gbps     = ((double)(a.n_cells * sizeof(Cell)) / 1e9) / (ns_total * 1e-9);
        read_ns_per_cell.push_back(ns_cell);
        read_gbps.push_back(gbps);
    }

    // measurements — WRITE mode
    std::vector<double> write_ns_per_cell; write_ns_per_cell.reserve(n_runs);
    std::vector<double> write_gbps;        write_gbps.reserve(n_runs);
    for (int r = 0; r < n_runs; ++r) {
        cpu_serialize();
        uint64_t t0 = TscClock::now();
        scan_write(a.data, a.n_cells, (uint16_t)r);
        uint64_t t1 = TscClock::now();
        cpu_serialize();
        double ns_total = clk.to_ns(t1 - t0);
        double ns_cell  = ns_total / (double)a.n_cells;
        double gbps     = ((double)(a.n_cells * sizeof(Cell)) / 1e9) / (ns_total * 1e-9);
        write_ns_per_cell.push_back(ns_cell);
        write_gbps.push_back(gbps);
    }

    auto sR = summarize(read_ns_per_cell);
    auto sW = summarize(write_ns_per_cell);
    auto bR = summarize(read_gbps);
    auto bW = summarize(write_gbps);

    // CSV on stdout
    std::printf("metric,n_cells,n_runs,pin_core,large_pages,median,mean,p95,p99,stdev,min,max\n");
    std::printf("read_ns_per_cell,%zu,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
        n_cells, n_runs, pin_core, a.used_lpages?1:0,
        sR.median, sR.mean, sR.p95, sR.p99, sR.stdev, sR.min, sR.max);
    std::printf("write_ns_per_cell,%zu,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
        n_cells, n_runs, pin_core, a.used_lpages?1:0,
        sW.median, sW.mean, sW.p95, sW.p99, sW.stdev, sW.min, sW.max);
    std::printf("read_gbps,%zu,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
        n_cells, n_runs, pin_core, a.used_lpages?1:0,
        bR.median, bR.mean, bR.p95, bR.p99, bR.stdev, bR.min, bR.max);
    std::printf("write_gbps,%zu,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
        n_cells, n_runs, pin_core, a.used_lpages?1:0,
        bW.median, bW.mean, bW.p95, bW.p99, bW.stdev, bW.min, bW.max);

    std::fprintf(stderr, "\n--- summary ---\n");
    std::fprintf(stderr, "read : median %.3f ns/cell  p95 %.3f  p99 %.3f  stdev %.3f  | %.2f GB/s\n",
        sR.median, sR.p95, sR.p99, sR.stdev, bR.median);
    std::fprintf(stderr, "write: median %.3f ns/cell  p95 %.3f  p99 %.3f  stdev %.3f  | %.2f GB/s\n",
        sW.median, sW.p95, sW.p99, sW.stdev, bW.median);

    free_tissue(a);
    (void)sink;
    return 0;
}
