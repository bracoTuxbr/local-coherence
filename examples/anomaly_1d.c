/* anomaly_1d.c — sliding-window anomaly detector using LC Runtime.
 *
 * Use case: 1D time-series (network flows/sec, log rate, etc).
 * Inject value at "now" position. Step adaptive. If active_count
 * spikes above baseline, alert.
 *
 * Build:
 *   gcc -std=c99 -O2 -I include -L build -o build/anomaly_1d examples/anomaly_1d.c -llc
 */

#include "lc/lc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WINDOW 8192
#define BASELINE_GENS 50
#define ALERT_THRESHOLD 80

int main(void) {
    lc_tissue_t* t = lc_create_1d(WINDOW);
    if (!t) { fprintf(stderr, "FAIL create\n"); return 1; }

    /* simulate 60 seconds of "normal" traffic (low-amplitude noise) */
    printf("Simulating 60s baseline (low noise)...\n");
    for (int sec = 0; sec < 60; ++sec) {
        size_t pos = (size_t)sec * 100;
        if (pos >= WINDOW) pos = pos % WINDOW;
        lc_inject_1d(t, pos, 1, 1500);  /* baseline pulse */
        lc_step(t, 5);
    }
    size_t baseline = lc_active_count(t);
    printf("Baseline active count: %zu\n\n", baseline);

    /* simulate burst: large amplitude pulse (DDoS-like) */
    printf("Injecting burst at sec 60...\n");
    lc_inject_1d(t, WINDOW / 2, 200, 60000);

    /* monitor for 10 seconds, report active_count */
    int alerted = 0;
    for (int sec = 60; sec < 70; ++sec) {
        lc_step(t, 5);
        size_t active = lc_active_count(t);
        printf("  sec=%2d active=%4zu", sec, active);
        if (!alerted && active > baseline + ALERT_THRESHOLD) {
            printf("  *** ALERT: anomaly detected ***");
            alerted = 1;
        }
        printf("\n");
    }

    lc_destroy(t);
    return alerted ? 0 : 2;  /* exit 0 if alerted (expected), else 2 */
}
