/* pulse_1d.c — minimal usage example of LC Runtime API.
 *
 * Build:
 *   gcc -std=c99 -Wall -Wextra -I include -L build -o build/pulse_1d examples/pulse_1d.c -llc
 *   (Windows: build directory must contain liblc.dll at runtime)
 *
 * Run:
 *   ./build/pulse_1d
 */

#include "lc/lc.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int abi_maj = 0, abi_min = 0;
    lc_abi_version(&abi_maj, &abi_min);
    printf("LC Runtime ABI %d.%d\n", abi_maj, abi_min);
    printf("Build: %s\n\n", lc_build_info());

    /* Create a 1M-cell 1D tissue. */
    const size_t n = 1u << 20;
    lc_tissue_t* t = lc_create_1d(n);
    if (!t) {
        fprintf(stderr, "FAIL: lc_create_1d returned NULL\n");
        return 1;
    }

    /* Inject single pulse at center. */
    lc_inject_1d(t, n / 2, 1, 60000);
    printf("After inject: active = %zu, r_eff = %zu\n",
           lc_active_count(t), lc_r_eff(t));

    /* Step 50 generations. */
    for (int batch = 1; batch <= 5; ++batch) {
        lc_step(t, 10);
        printf("After %d gens: active = %4zu, r_eff = %3zu\n",
               batch * 10, lc_active_count(t), lc_r_eff(t));
    }

    /* List dirty chunks. */
    size_t dirty_idx[32];
    size_t n_dirty = lc_active_chunks(t, dirty_idx, 32);
    printf("\nDirty chunks (%zu):", n_dirty);
    for (size_t i = 0; i < n_dirty && i < 16; ++i) {
        printf(" %zu", dirty_idx[i]);
    }
    printf(n_dirty > 16 ? " ...\n" : "\n");

    /* Run until bit-exact stable. */
    int gen_stable = lc_step_until_stable(t, 5000);
    printf("\nStabilized at gen %d\n", gen_stable);
    printf("Final active = %zu, r_eff = %zu\n",
           lc_active_count(t), lc_r_eff(t));

    lc_destroy(t);
    printf("\nOK\n");
    return 0;
}
