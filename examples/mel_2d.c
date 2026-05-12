/* mel_2d.c — 2D tissue (mel x time) propagation example. */

#include "lc/lc.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const size_t H = 80;
    const size_t W = 256;

    printf("step 1: create_2d %zux%zu\n", H, W);
    fflush(stdout);
    lc_tissue_t* t = lc_create_2d(H, W);
    if (!t) { fprintf(stderr, "FAIL create_2d\n"); return 1; }

    printf("step 2: inject 30x140 cells\n"); fflush(stdout);
    for (size_t r = 20; r < 50; ++r) {
        for (size_t c = 60; c < 200; ++c) {
            uint16_t v = (uint16_t)(30000 - 200 * (r > 35 ? r - 35 : 35 - r));
            lc_inject_2d(t, r, c, v);
        }
    }
    printf("after inject: active = %zu\n", lc_active_count(t)); fflush(stdout);

    printf("step 3: step_adaptive 10 gens\n"); fflush(stdout);
    lc_step_adaptive(t, 10);
    printf("after 10 adaptive: active = %zu\n", lc_active_count(t));
    fflush(stdout);

    printf("step 4: more 40 adaptive\n"); fflush(stdout);
    lc_step_adaptive(t, 40);
    printf("after 50 adaptive: active = %zu\n", lc_active_count(t));
    fflush(stdout);

    printf("step 5: get_field\n"); fflush(stdout);
    size_t total = H * W;
    uint16_t* out = (uint16_t*)malloc(total * sizeof(uint16_t));
    if (!out) { lc_destroy(t); return 2; }
    size_t copied = lc_get_field(t, out, total);
    printf("copied %zu cells\n", copied); fflush(stdout);

    printf("step 6: destroy\n"); fflush(stdout);
    free(out);
    lc_destroy(t);
    printf("OK\n");
    return 0;
}
