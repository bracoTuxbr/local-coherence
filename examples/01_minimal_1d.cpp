// 01_minimal_1d.cpp — the smallest meaningful demo of Local Coherence.
//
// Builds a 1024-cell tissue, injects a pulse at the centre, steps it for
// 100 generations under the canonical kernel, and prints the active set
// after each step. Demonstrates the Manhattan-diamond growth predicted
// by the Differential Activity Principle (PAD): |A(t)| = 2t + 1 in 1D,
// until decay drives the front below uint16 representable.
//
// Build (Windows MinGW):   g++ -O2 -std=c++17 examples/01_minimal_1d.cpp -Isrc -o min1d.exe
// Build (Linux gcc):       g++ -O2 -std=c++17 examples/01_minimal_1d.cpp -Isrc -o min1d
// Run:                     ./min1d

#include "tissue.hpp"
#include "propagate.hpp"
#include <cstdio>
#include <cstdint>

int main() {
    constexpr size_t N = 1024;
    constexpr int STEPS = 100;
    constexpr uint16_t INJECT = 30000;

    lc::Tissue1D tissue(N);
    lc::HotField16 hot(tissue);

    // Inject a single pulse at centre
    hot[N / 2] = INJECT;
    printf("step 0  : peak=%u  active=%zu  (injected)\n", (unsigned)hot[N / 2], (size_t)1);

    for (int t = 1; t <= STEPS; ++t) {
        lc::propagate_1d_canonical(hot, N, 1);

        // Count active cells
        size_t active = 0;
        uint16_t peak = 0;
        for (size_t i = 0; i < N; ++i) {
            if (hot[i] > 0) {
                ++active;
                if (hot[i] > peak) peak = hot[i];
            }
        }

        if (t <= 10 || t % 10 == 0) {
            // Predicted Manhattan-diamond: 2t+1 (while alive)
            size_t predicted = 2 * t + 1;
            printf("step %3d: peak=%5u  active=%4zu  predicted=%4zu  %s\n",
                   t, (unsigned)peak, active, predicted,
                   (active == predicted) ? "MATCH" : "decay");
        }
    }

    return 0;
}
