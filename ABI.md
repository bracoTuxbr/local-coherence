# LC Runtime ABI v1.2

Public C99 API for the Local Coherence Runtime, declared in `include/lc/lc.h`.

This document specifies pre/post-conditions, thread-safety, and stability
contract of every public symbol. Once tagged v1.0.0, breaking changes
require a major version bump.

## Versioning

`LC_ABI_MAJOR.LC_ABI_MINOR` macros in `lc.h`. Currently **1.2**.

- **Major bump**: incompatible signature change, removal, or behavior break
- **Minor bump**: additive only (new function appended, new flag, new opaque field)

Use `lc_abi_version(int*, int*)` at runtime to detect mismatch.

### Version history

| ABI | Release | Added |
|---|---|---|
| 1.0 | 2026-05-08 | First public surface (16 functions, lifecycle / inject / step / observe) |
| 1.1 | 2026-05-09 | `lc_set_sig_delta(t, sd)` — runtime-tunable significance threshold; default 4 (preserves all v1.0 EXACT golden numbers). `sig_delta=1` makes the dirty path bit-exact with a naive (no-freeze) reference. |
| 1.2 | 2026-05-10 | `lc_set_kernel(t, kernel_id)` — pluggable kernel selection at runtime. Four kernels exposed: `LC_KERNEL_CANONICAL` (default, bit-exact baseline), `LC_KERNEL_SIMPLE_AVG`, `LC_KERNEL_EMA`, `LC_KERNEL_CANONICAL_U64` (uint64 variant for hash-collision workloads). Underlying `HotField<T>` generalization to `uint8_t / uint16_t / uint32_t` cell types. |

## Build artifacts

- `liblc.dll` (Windows MinGW/MSVC) or `liblc.so` (Linux)
- `liblc.a` (Windows import library) or no separate import on Linux
- `include/lc/lc.h` (single public header, C99 + C++ compatible via `extern "C"`)

Size budget: liblc <500 KB. Currently ~55 KB on Windows GCC 16.1.0 -O3.

## Thread-safety

| operation | per-tissue | inter-tissue |
|---|---|---|
| `lc_create_*`, `lc_destroy` | n/a | safe |
| `lc_set_threads`, `lc_set_pulse_value` | NOT safe (must precede first step) | safe |
| `lc_inject_*` | NOT safe (mutates) | safe |
| `lc_step*` | NOT safe (mutates) | safe |
| `lc_active_count`, `lc_r_eff`, `lc_get_field`, `lc_active_chunks` | safe to call concurrent with other reads on same tissue; NOT safe concurrent with step/inject | safe |
| `lc_abi_version`, `lc_build_info` | safe | safe |

The runtime does not lock internally. Caller is responsible for synchronizing
operations on the same tissue handle. Different tissues can be operated by
different threads without coordination.

## Determinism

Given identical sequence of `lc_inject_*` and `lc_step*` calls on a tissue
created with the same dimensions, the resulting field is bit-exact reproducible
across runs and across operating systems. Validated against 17 golden numbers
in `benchmarks/golden_numbers.txt`.

The kernel is integer fixed-point (`uint16_t`); no floating point is used in
the propagation rule.

## Function specifications

### `lc_tissue_t* lc_create_1d(size_t n_cells)`

**Pre:** `n_cells > 0` and `n_cells * sizeof(uint16_t) ≤ SIZE_MAX/2`.

**Post:** returns opaque handle to a 1D tissue with all cells initialized to 0,
all chunks marked clean (no dirty bits), thread count = 1, default pulse =
60000.

**Failure:** returns `NULL` on allocation failure or invalid `n_cells`.

**Cost:** O(n_cells) for zeroing.

---

### `lc_tissue_t* lc_create_2d(size_t H, size_t W)`

**Pre:** `H > 0`, `W > 0`, `H * W * sizeof(uint16_t)` does not overflow.

**Post:** returns 2D tissue, row-major layout, halo perimetral. Tiles of
`TILE_R × TILE_C = 64 × 64` cells. All cells = 0, all tiles clean.

**Failure:** `NULL` on alloc failure or invalid dimensions.

---

### `lc_tissue_t* lc_create_from_buffer_1d(const uint16_t* buf, size_t n)`

**Pre:** `buf != NULL`, `n > 0`.

**Post:** returns 1D tissue with cells [0..n) initialized from `buf`. All
chunks marked dirty (so first `lc_step` processes everything). Caller retains
ownership of `buf`; `lc_destroy()` will NOT free it.

**Note:** v1.0 copies `buf` into internal allocator-owned memory (does NOT
share). Future v1.1 may add `lc_create_view` for zero-copy.

---

### `void lc_destroy(lc_tissue_t* t)`

**Pre:** `t` is a valid handle from `lc_create_*` OR `t == NULL`.

**Post:** all internal resources freed. After return, `t` must not be used.

**Idempotent:** safe to call on `NULL`.

---

### `void lc_set_threads(lc_tissue_t* t, int n)`

**Pre:** `t != NULL`, `n >= 1`. Must be called BEFORE first `lc_step*`.

**Post:** records thread count for future MT execution. **In v1.0 this is
ignored — the runtime executes single-threaded.** Multi-thread integration
(M19/M20) is planned for v1.1.

---

### `void lc_set_pulse_value(lc_tissue_t* t, uint16_t v)`

**Pre:** `t != NULL`.

**Post:** sets the default pulse value used internally. Currently no public
function uses this default — `lc_inject_*` always takes explicit `v`.
Reserved for future API additions.

---

### `void lc_set_sig_delta(lc_tissue_t* t, int sd)`  *(ABI 1.1)*

**Pre:** `t != NULL`, `sd >= 0`.

**Post:** sets the significance threshold for the dirty bitmap. A chunk
is considered "dirty" only when its per-cell delta exceeds `sd`. Default
is 4 (preserves every v1.0 EXACT golden number). Set to 1 to make the
dirty path bit-exact with a naive (no-freeze) reference; set higher to
skip more chunks at the cost of L1 distance accuracy. Thread-safety:
must be called before the first `lc_step*`.

---

### `void lc_set_kernel(lc_tissue_t* t, int kernel_id)`  *(ABI 1.2)*

**Pre:** `t != NULL`, `kernel_id` is one of the constants below.

**Post:** selects the propagation rule applied per step. Default is
`LC_KERNEL_CANONICAL`.

| `kernel_id` | rule |
|---|---|
| `LC_KERNEL_CANONICAL` | `y[i] = ((x[i-1] + 2*x[i] + x[i+1]) >> 2) * 255/256` (default, bit-exact baseline) |
| `LC_KERNEL_SIMPLE_AVG` | `y[i] = (x[i-1] + x[i] + x[i+1]) / 3` (no decay) |
| `LC_KERNEL_EMA` | `y[i] = ((x[i-1] + 2*x[i] + x[i+1]) >> 2) * 254/256` (faster decay) |
| `LC_KERNEL_CANONICAL_U64` | uint64-input variant of canonical (for hash-collision counters) |

Switching kernels changes the field evolution and therefore the golden
numbers — only `LC_KERNEL_CANONICAL` preserves the v1.0 EXACT values.
Thread-safety: must be called before the first `lc_step*`.

---

### `void lc_inject_1d(lc_tissue_t* t, size_t pos, size_t len, uint16_t v)`

**Pre:** `t != NULL`, tissue is 1D. `pos + len` may exceed tissue length;
behavior is to clip to tissue length.

**Post:** cells `[pos, min(pos+len, n))` are set to `v`. Chunks containing
those cells, plus the immediate neighbors at chunk boundaries, are marked
dirty. Stability counters reset to 0 in the affected range.

**Cost:** O(len) for cell writes + O(n_chunks_affected/64) for bitmap update.

---

### `void lc_inject_2d(lc_tissue_t* t, size_t row, size_t col, uint16_t v)`

**Pre:** `t != NULL`, tissue is 2D, `row < H`, `col < W`.

**Post:** cell at (row, col) is set to `v`, tile containing it is marked
dirty.

**Out-of-range:** silently no-op (does not abort).

---

### `void lc_step(lc_tissue_t* t, int n_gens)`

**Pre:** `t != NULL`, `n_gens > 0`.

**Post:** advances `n_gens` generations using the canonical kernel `(l + 2c +
r) >> 2 * 255/256` (1D) or `(up + dn + lf + rt + 4c) >> 3 * 255/256` (2D),
processing only dirty chunks/tiles. Cells satisfying `|delta| <
SIGNIFICANT_DELTA` for `STABLE_THRESH` consecutive gens are frozen (chunk
removed from active set; reactivated when neighbor perturbs).

**Best for:** sparse perturbations (locally-structured input). Speedup up to
600× over a dense kernel on 1M cells with single pulse (M4 result).

**Performance regimes** (Ryzen 5 7520U single-thread):
- DRAM sparse: ~0.18 ns/cell effective with adaptive
- DRAM uniform: ~0.55 ns/cell (overhead exceeds savings)
- L2-fit: traditional float32 baseline outperforms by ~2.5×

---

### `void lc_step_adaptive(lc_tissue_t* t, int n_gens)`

**Pre:** same as `lc_step`. **2D only in v1.0**; for 1D, falls back to `lc_step`.

**Post:** same effect as `lc_step` but with the M5.5 dense/dirty switching
detector. Probe runs every 16 gens; switches to MODE_NAIVE when `|A|/|Λ| >
40%`, back to MODE_DIRTY when `< 25%`. Hysteresis prevents thrashing.

**Best for:** workloads with unknown/varying regime (mel-spectrogram, mixed
sparse/dense traffic).

**Validation:** robust on 10/10 real audio samples (M25), with 1 mode
transition per sample.

---

### `int lc_step_until_stable(lc_tissue_t* t, int max_gens)`

**Pre:** `t != NULL`, **1D tissue only** in v1.0, `max_gens > 0`.

**Post:** runs `propagate_1d` (NOT `lc_step`) generation by generation until
`memcmp(prev, next, n)` returns 0 or `max_gens` is hit.

**Returns:** generation at which bit-exact stability is reached, or -1 if
not within `max_gens`.

**Note:** uses dense kernel (no dirty bitmap). For 1M cells, typical
stabilization in 460–510 gens for single pulse (M-3 result).

---

### `size_t lc_active_count(const lc_tissue_t* t)`

**Pre:** `t != NULL`.

**Post:** returns count of cells with value > 0 in the current "live"
buffer (after most recent step).

**Cost:** O(n_cells). For frequent monitoring, prefer counting via
`lc_active_chunks` which is O(n_chunks/64).

---

### `size_t lc_r_eff(const lc_tissue_t* t)`

**Pre:** `t != NULL`, **1D tissue only**. Tissue should have a perturbation
near the center (cell `n/2`) — for arbitrary patterns, result is undefined.

**Post:** returns effective propagation radius from center: max `k` such
that `data[center+k] > 0` OR `data[center-k] > 0`.

---

### `size_t lc_get_field(const lc_tissue_t* t, uint16_t* out, size_t max_n)`

**Pre:** `t != NULL`, `out != NULL`, `max_n` allows at least one cell.

**Post:** copies up to `max_n` cells of the live buffer into `out`. For 2D
tissues, copies row-major (without halo padding).

**Returns:** number of cells written = `min(n_tissue, max_n)`.

---

### `size_t lc_active_chunks(const lc_tissue_t* t, size_t* out, size_t max)`

**Pre:** `t != NULL`, `out != NULL`.

**Post:** writes indices of dirty chunks (1D) or dirty tiles (2D) into
`out[0..count)`. Each chunk index `c` covers cells `[c*64, (c+1)*64)` in 1D,
or tile `(c / NTC, c % NTC)` of 64×64 cells in 2D.

**Returns:** number of indices written = `min(n_dirty, max)`.

**Cost:** O(n_words/64).

---

### `void lc_abi_version(int* major, int* minor)`

**Pre:** `major` and/or `minor` may be `NULL` (caller may want only one).

**Post:** writes `LC_ABI_MAJOR` and `LC_ABI_MINOR` from the compiled library.

**Use:** runtime version check before calling other API.

---

### `const char* lc_build_info(void)`

**Pre:** none.

**Post:** returns static C string with build identifier (compile date, time,
version label). Lifetime is the library's; do not free.

## Performance contract (informational, not normative)

On AMD Ryzen 5 7520U Zen 2 (15 W TDP), GCC 16.1.0 `-O3 -mavx2`:

- `lc_step` 1D, 1M cells, sparse pulse: ~0.0007 ns/cell effective (M4)
- `lc_step_adaptive` 2D, 2048×2048 sparse: ~0.18 ns/cell (M26)
- `lc_step` 2D, 2048×2048 uniform: ~0.55 ns/cell (M26)
- DRAM bandwidth saturation: ~10 GB/s (M-roofline e17)

These are reference values, not guarantees. Tolerance ±25% per regression
suite default.

## Known limitations

- v1.0 is single-threaded internally. `lc_set_threads(t, >1)` is recorded
  but not honored. Multi-thread (M19/M20) lands in v1.1.
- `lc_step_adaptive` is 2D-only. 1D adaptive is planned for v1.1.
- `lc_step_until_stable` is 1D-only. 2D stable detection requires per-tile
  comparison and is deferred.
- Long-range correlation between cells in distance > ~32 cells is not
  detectable by the current kernel due to integer quantization decay (M24).
  Applications requiring this need: continuous source repaint, hierarchical
  multi-level (planned v2), or external decoder.

## Source files

```
include/lc/lc.h              public C99 API header
src/                         runtime implementation (15 headers + lc.cpp)
examples/01_minimal_1d.cpp   Manhattan-diamond growth demo (C++)
examples/pulse_1d.c          single-pulse propagation in 1D (C)
examples/anomaly_1d.c        sliding-window anomaly detector (C)
examples/mel_2d.c            2D propagation demo (C)
tests/test_core.cpp          19 unit tests / 63 assertions
benchmarks/                  paper experiment drivers + golden_numbers.txt
tools/build.ps1              build script (w64devkit / mingw)
tools/regression_test.ps1    regression gate against golden_numbers.txt
ABI.md                       this file
docs/embedding-guide.md      tutorial-level usage guide
docs/architecture.md         runtime / application boundary
```

`liblc.dll` (Windows) and `liblc.so` (Linux) are produced under
`build/` after running `tools/build.ps1`. They are not committed to the
repository (`build/` is `.gitignored`).
