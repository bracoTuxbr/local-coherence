// e16.cpp — M-3, M-4: time to stabilization + active region trace
//
// Question M-3: given a central pulse in uint16, how many gens until the system
// returns to zero (or stops changing)?
// Question M-4: how do |A(t)| and r_eff(t) evolve over time?
//
// Theoretical hypothesis: for pulse v0 with decay (255/256), v_max(t) ~ v0 * (255/256)^t
// at the center, but combined with (1/4)^d at distance d. Total time t* such that
// v_max(t*) < 1 in uint16:
//   v0 * (255/256)^t < 1 -> t > log(v0) / log(256/255) = log(60000) / 0.00390625
//   t* ~= log(60000) * 256 ~ 2820 gens
// But decay (1/4) by distance swallows the front earlier at ~r* = 8 gens.
// After that, the system is left with a "plateau" of central cells decaying by (255/256).

#include "runtime.hpp"
#include "propagate.hpp"
#include "metrics.hpp"

#include <cstdio>
#include <cstdlib>

using namespace cl;

int main(int argc, char** argv) {
    int max_gens = (argc > 1) ? std::atoi(argv[1]) : 5000;
    uint16_t pulse = (argc > 2) ? (uint16_t)std::atoi(argv[2]) : 60000;

    auto info = detect_cpu();
    std::fprintf(stderr, "=== local-coherence / E16 M-3/M-4 stabilization + trace ===\n");
    print_cpu(info);
    std::fprintf(stderr, "max_gens=%d pulse=%u\n\n", max_gens, pulse);

    boost_process_priority();
    pin_thread_to_core(0);
    boost_thread_priority();

    const size_t n = 1u << 18; // 256K
    auto a = alloc_hf16(n, 1);
    auto b = alloc_hf16(n, 1);
    if (!a.base || !b.base) {
        std::fprintf(stderr, "ERROR: alloc failed\n");
        return 1;
    }
    for (size_t i = 0; i < n; ++i) a.data[i] = 0;
    a.data[n/2] = pulse;
    size_t center = n/2;

    // === M-3: time to stabilization ===
    std::fprintf(stderr, "--- M-3: time to stabilization ---\n");
    auto stab = run_until_no_change_1d(a, b, max_gens, center);
    std::fprintf(stderr, "  gen_stable        = %d\n", stab.gen_stable);
    std::fprintf(stderr, "  gens_with_activity= %zu\n", stab.gens_with_activity);
    std::fprintf(stderr, "  peak_active       = %zu (gen %d)\n", stab.peak_active, stab.peak_gen);
    std::fprintf(stderr, "  peak_r_eff        = %zu\n", stab.peak_r_eff);

    // === M-4: trace ===
    // Needs clean buffers. Re-alloc.
    free_hf16(a); free_hf16(b);
    a = alloc_hf16(n, 1);
    b = alloc_hf16(n, 1);
    if (!a.base || !b.base) { return 1; }
    for (size_t i = 0; i < n; ++i) a.data[i] = 0;
    a.data[n/2] = pulse;

    std::fprintf(stderr, "\n--- M-4: trace |A(t)| e r_eff(t) ---\n");
    int trace_gens = (stab.gen_stable > 0) ? stab.gen_stable : 200;
    if (trace_gens > 200) trace_gens = 200;  // cap for log
    auto tr = propagation_trace_1d(a, b, trace_gens, center);
    std::printf("gen,active,r_eff\n");
    std::fprintf(stderr, "  gen     |A|   r_eff\n");
    int log_step = 1;
    if (trace_gens > 50) log_step = trace_gens / 20;
    for (int g = 0; g <= trace_gens; g += log_step) {
        if ((size_t)g >= tr.active.size()) break;
        std::fprintf(stderr, "  %4d  %5zu  %5zu\n", g, tr.active[g], tr.r_eff[g]);
        std::printf("%d,%zu,%zu\n", g, tr.active[g], tr.r_eff[g]);
    }
    // ensure print of the last gen even if log_step skips it
    if (trace_gens % log_step != 0 && (size_t)trace_gens < tr.active.size()) {
        std::fprintf(stderr, "  %4d  %5zu  %5zu\n",
                     trace_gens, tr.active[trace_gens], tr.r_eff[trace_gens]);
        std::printf("%d,%zu,%zu\n",
                    trace_gens, tr.active[trace_gens], tr.r_eff[trace_gens]);
    }

    // === Invariant checks ===
    int fail = 0;
    if (stab.gen_stable < 0 || stab.gen_stable > max_gens) {
        std::fprintf(stderr, "FAIL: did not converge in %d gens\n", max_gens);
        ++fail;
    }
    if (stab.peak_active < 15) {
        std::fprintf(stderr, "FAIL: peak_active=%zu < 15 (expected >= 2t+1 with t>=7)\n",
                     stab.peak_active);
        ++fail;
    }
    if (stab.peak_r_eff < 7) {
        std::fprintf(stderr, "FAIL: peak_r_eff=%zu < 7 (expected >= log(60000)/log(4))\n",
                     stab.peak_r_eff);
        ++fail;
    }

    std::fprintf(stderr, "\n=== summary ===\n");
    std::fprintf(stderr, "fail=%d\n", fail);
    if (fail == 0) {
        std::fprintf(stderr, "PASS: M-3/M-4 confirmed\n");
        std::fprintf(stderr, "  System converges in %d gens (bit-exact definition)\n",
                     stab.gen_stable);
        std::fprintf(stderr, "  Front reaches radius %zu before decay swallows it\n",
                     stab.peak_r_eff);
    }

    free_hf16(a); free_hf16(b);
    return fail == 0 ? 0 : 1;
}
