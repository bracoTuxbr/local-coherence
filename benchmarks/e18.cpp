// e18.cpp — dense trace for paper figures (DOES NOT validate, only produces CSV)
//
// Prints |A(t)| 1D and 2D point by point, without sampling. Output to stdout in
// wide CSV format: gen,a1d,r1d,a2d
//
// Usage: e18 <max_gens> <pulse>
//
// Does not touch golden numbers. e15/e16 remain authoritative.

#include "runtime.hpp"
#include "propagate.hpp"
#include "tissue2d.hpp"
#include "diffuse2d.hpp"
#include "metrics.hpp"

#include <cstdio>
#include <cstdlib>

using namespace cl;

int main(int argc, char** argv) {
    int max_gens = (argc > 1) ? std::atoi(argv[1]) : 50;
    uint16_t pulse = (argc > 2) ? (uint16_t)std::atoi(argv[2]) : 60000;

    auto info = detect_cpu();
    std::fprintf(stderr, "=== local-coherence / E18 dense trace (paper figs) ===\n");
    print_cpu(info);
    std::fprintf(stderr, "max_gens=%d pulse=%u\n", max_gens, pulse);

    boost_process_priority();
    pin_thread_to_core(0);
    boost_thread_priority();

    // 1D trace
    const size_t n1d = 1u << 18;
    auto a1 = alloc_hf16(n1d, 1);
    auto b1 = alloc_hf16(n1d, 1);
    if (!a1.base || !b1.base) { std::fprintf(stderr, "FAIL alloc 1D\n"); return 1; }
    for (size_t i = 0; i < n1d; ++i) a1.data[i] = 0;
    a1.data[n1d/2] = pulse;
    auto tr1 = propagation_trace_1d(a1, b1, max_gens, n1d/2);
    free_hf16(a1); free_hf16(b1);

    // 2D trace
    const size_t H = 256, W = 256;
    auto a2 = alloc_hf2d(H, W, 1);
    auto b2 = alloc_hf2d(H, W, 1);
    if (!a2.base || !b2.base) { std::fprintf(stderr, "FAIL alloc 2D\n"); return 1; }
    clear_hf2d(a2); clear_hf2d(b2);
    a2.data[(H/2) * a2.stride + (W/2)] = pulse;
    auto tr2 = propagation_trace_2d(a2, b2, max_gens);
    free_hf2d(a2); free_hf2d(b2);

    std::printf("gen,a1d,r1d,a2d\n");
    for (int g = 0; g <= max_gens; ++g) {
        size_t a1d = (g < (int)tr1.active.size()) ? tr1.active[g] : 0;
        size_t r1d = (g < (int)tr1.r_eff.size()) ? tr1.r_eff[g] : 0;
        size_t a2d = (g < (int)tr2.active.size()) ? tr2.active[g] : 0;
        std::printf("%d,%zu,%zu,%zu\n", g, a1d, r1d, a2d);
    }
    return 0;
}
