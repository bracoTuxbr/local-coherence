// 01_minimal_1d.cpp — the smallest meaningful demo of Local Coherence,
// using only the public C99 API exposed in include/lc/lc.h.
//
// Builds a 1024-cell tissue, injects a pulse at the centre, steps it for
// 100 generations under the canonical kernel, and prints field stats
// after each step. The active region grows by one cell per generation
// (Manhattan diamond, |A(t)| = 2t+1 in 1D) until decay drives the front
// below uint16 representable.
//
// Build (Linux gcc, dynamic):
//   gcc -O2 -std=c99 -I include examples/01_minimal_1d.cpp -L build -llc -o min1d
// Build (Windows MinGW, dynamic):
//   gcc -O2 -std=c99 -I include examples/01_minimal_1d.cpp -L build -llc -o min1d.exe
// Run (Linux):   LD_LIBRARY_PATH=build ./min1d
// Run (Windows): PATH=build;%PATH%  &&  min1d.exe

#include "lc/lc.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main(void) {
    const size_t   N      = 1024;
    const int      STEPS  = 100;
    const uint16_t INJECT = 30000;

    lc_tissue_t* t = lc_create_1d(N);
    if (!t) {
        fprintf(stderr, "lc_create_1d failed\n");
        return 1;
    }
    lc_set_kernel(t, LC_KERNEL_CANONICAL);

    lc_inject_1d(t, N / 2, 1, INJECT);
    printf("step   0: peak=%5u  active=%4zu  predicted=   1\n",
           (unsigned)INJECT, (size_t)lc_active_count(t));

    uint16_t* field = (uint16_t*)malloc(N * sizeof(uint16_t));
    if (!field) { lc_destroy(t); return 1; }

    for (int gen = 1; gen <= STEPS; ++gen) {
        lc_step(t, 1);
        size_t n_read = lc_get_field(t, field, N);
        uint16_t peak = 0;
        size_t   active = 0;
        for (size_t i = 0; i < n_read; ++i) {
            if (field[i] > 0) {
                ++active;
                if (field[i] > peak) peak = field[i];
            }
        }
        if (gen <= 10 || gen % 10 == 0) {
            size_t predicted = (size_t)(2 * gen + 1);
            const char* tag = (active == predicted) ? "MATCH" : "decay";
            printf("step %3d: peak=%5u  active=%4zu  predicted=%4zu  %s\n",
                   gen, (unsigned)peak, active, predicted, tag);
        }
    }

    free(field);
    lc_destroy(t);
    return 0;
}
