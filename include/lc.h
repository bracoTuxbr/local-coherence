/* lc.h — Local Coherence Runtime, public C99 API.
 *
 * Embeddable runtime for sparse local propagation on commodity CPUs.
 * Tissue layout: contiguous SoA uint16. Operating point: tissues > 1M cells
 * with locally-structured input (sparse perturbations or temporal stability).
 *
 * Thread-safety:
 *   - Functions are NOT thread-safe per-tissue.
 *   - Different tissues may be operated by different threads.
 *   - Internal multi-thread (when lc_set_threads > 1) is handled by the runtime.
 *
 * Determinism:
 *   - All operations are bit-exact reproducible given identical seeds + injections.
 *   - The kernel is integer fixed-point (uint16). No float. No undefined order.
 *
 * License: Apache 2.0 (planned for v1.0 public release).
 */

#ifndef LC_H
#define LC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version. Bumped on incompatible changes to this header.
 * MINOR bumps add new symbols but keep old ones binary-compatible. */
#define LC_ABI_MAJOR 1
#define LC_ABI_MINOR 2

/* Kernel selection (ABI 1.2+). Each kernel implements a different stencil
 * rule for cell propagation. Default is LC_KERNEL_CANONICAL (bit-exact
 * with all golden_numbers.txt baselines). */
typedef enum {
    LC_KERNEL_CANONICAL  = 0,  /* (l + 2c + r) >> 2 * 255/256 — default */
    LC_KERNEL_SIMPLE_AVG = 1,  /* (l + c + r) / 3 — NO decay */
    LC_KERNEL_EMA        = 2   /* peso forte centro + decay 254/256 */
} lc_kernel_id_t;

/* Opaque tissue handle. */
typedef struct lc_tissue lc_tissue_t;

/* ---------------------------------------------------------------------
 * Lifecycle
 * --------------------------------------------------------------------- */

/* Create a 1D tissue of n_cells. Returns NULL on allocation failure. */
lc_tissue_t* lc_create_1d(size_t n_cells);

/* Create a 2D tissue of HxW cells. Returns NULL on allocation failure. */
lc_tissue_t* lc_create_2d(size_t H, size_t W);

/* Create a 1D tissue from external buffer. Caller retains ownership of buf;
 * lc_destroy() will NOT free buf. n is in cells (uint16_t units). */
lc_tissue_t* lc_create_from_buffer_1d(const uint16_t* buf, size_t n);

/* Free a tissue. Safe to call on NULL. After return, handle is invalid. */
void lc_destroy(lc_tissue_t* t);

/* ---------------------------------------------------------------------
 * Configuration (must be called BEFORE first lc_step on a tissue)
 * --------------------------------------------------------------------- */

/* Number of worker threads. n=1 forces single-thread (default).
 * Effective only on tissues > L3-fit (~1M cells). Pinned to physical cores. */
void lc_set_threads(lc_tissue_t* t, int n);

/* Default value used by lc_inject_* when value is omitted. */
void lc_set_pulse_value(lc_tissue_t* t, uint16_t v);

/* Select kernel for propagation (1D only; no-op on 2D).
 * Default is LC_KERNEL_CANONICAL — bit-exact with golden_numbers baselines.
 * Other kernels change the stencil rule for the cell update.
 * Available since ABI 1.2. */
void lc_set_kernel(lc_tissue_t* t, lc_kernel_id_t id);

/* Set significance threshold for freeze logic (1D only; no-op on 2D).
 * Cells with |new-old| < sig_delta count as "stable" → contribute to chunk
 * being marked clean. Default = 4 (canonical SIGNIFICANT_DELTA from M0).
 * Trade-off:
 *   - sig_delta=1: dirty path is BIT-EXACT with naive, same correctness, full speedup.
 *   - sig_delta=4: canonical golden_numbers regime; small residual divergence.
 *   - sig_delta=16+: more aggressive freeze, higher speedup, larger divergence.
 * Available since ABI 1.1. */
void lc_set_sig_delta(lc_tissue_t* t, uint16_t v);

/* ---------------------------------------------------------------------
 * Injection (mark cells as active perturbations)
 * --------------------------------------------------------------------- */

/* Inject value v into [pos, pos+len) of a 1D tissue.
 * Marks affected chunks as dirty. */
void lc_inject_1d(lc_tissue_t* t, size_t pos, size_t len, uint16_t v);

/* Inject value v at (row, col) of a 2D tissue. */
void lc_inject_2d(lc_tissue_t* t, size_t row, size_t col, uint16_t v);

/* ---------------------------------------------------------------------
 * Step (advance N generations)
 * --------------------------------------------------------------------- */

/* Run n_gens iterations of the canonical kernel + dirty bitmap + freeze.
 * Best for sparse perturbations. */
void lc_step(lc_tissue_t* t, int n_gens);

/* Run n_gens with adaptive dense/dirty switching (M5.5).
 * Best when workload regime is unknown a priori. Probes every 16 gens
 * and switches between MODE_NAIVE (>40% dirty) and MODE_DIRTY (<25% dirty)
 * with hysteresis. */
void lc_step_adaptive(lc_tissue_t* t, int n_gens);

/* Run until prev[]==next[] bit-exact (system stable) or max_gens reached.
 * Returns gen at which stabilization occurred, or -1 if max_gens hit. */
int lc_step_until_stable(lc_tissue_t* t, int max_gens);

/* ---------------------------------------------------------------------
 * Observation (non-destructive read-only)
 * --------------------------------------------------------------------- */

/* Number of cells with value > 0. Constant-time invariant: |A(t)|. */
size_t lc_active_count(const lc_tissue_t* t);

/* Effective radius: max k such that data[center+k] > 0 OR data[center-k] > 0.
 * For 1D tissues with central perturbation. Returns 0 if tissue empty. */
size_t lc_r_eff(const lc_tissue_t* t);

/* Copy current field (live buffer) into out[0..max_n).
 * Returns number of cells actually written (min(n_tissue, max_n)).
 * Does not modify tissue. */
size_t lc_get_field(const lc_tissue_t* t, uint16_t* out, size_t max_n);

/* List indices of dirty chunks. Returns count written into out[0..max).
 * Each chunk index covers cells [c*64 .. (c+1)*64). */
size_t lc_active_chunks(const lc_tissue_t* t, size_t* out, size_t max);

/* ---------------------------------------------------------------------
 * Diagnostics
 * --------------------------------------------------------------------- */

/* Returns ABI version compiled into this library.
 * Caller compares against LC_ABI_MAJOR/MINOR macros to detect mismatch. */
void lc_abi_version(int* major, int* minor);

/* Returns library build identifier (compile-time). Useful for bug reports. */
const char* lc_build_info(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LC_H */
