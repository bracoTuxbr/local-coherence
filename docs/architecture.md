# Architecture

This document describes the runtime / application boundary, the public C API
surface, and the rules of the road for embedding `liblc` in larger systems.

The **paradigm** itself (the Differential Activity Principle, propagation
rules, freeze semantics) is explained in [`paradigm.md`](paradigm.md). This
document is operational: it answers *how the code is organized* and
*what belongs inside the runtime vs outside*.

## What the runtime is

CPU-native engine for **local propagation + sparse execution** over a
continuous tissue in memory.

Category: runtime / engine (think SQLite, llama.cpp, BEAM). Not a library
of point calls. Not a framework that drives flow.

Job: execute local propagation + freeze + adaptive switching +
coordinated multi-thread, absurdly well, over uint16 1D/2D tissues.

## Boundary principle

The runtime is responsible for **propagation primitives + non-destructive
observation**. The application is responsible for **problem semantics**.

### Intrinsic to the runtime

- Cells contiguous in memory, halo, cache-friendly SoA layout
- Local kernel (canonical default; pluggable via `lc_set_kernel`)
- Chunked dirty bitmap + stability-based freeze
- Adaptive runtime (dense ↔ dirty switching with hysteresis)
- Multi-thread with sense-reversing barrier or wave-driven cv-parking
- Cross-CCD coherence (when multi-CCD hardware is available)
- Metrics: `active_count`, `r_eff`, `propagation_trace`, cycles/cell, GB/s

### Outside the runtime (application's job)

- **Network I/O** (AF_XDP, NIC, eBPF) → application reads packet,
  extracts feature, calls `lc_inject`
- **Disk I/O** (mmap, snapshot, replay) → application does mmap, passes
  pointer to `lc_create_from_buffer`
- **Protocol parsing** (DNS, BGP, NATS pub/sub) → 100% application
- **Audio frontend** (FFT, mel-spec, WAV) → application computes mel,
  injects into the runtime
- **Downstream classifiers** (LOO, nearest centroid, MLP) → application
  reads metrics + classifies
- **Learned ML models** → application trains offline, embeds weights as
  a table read by the rule

## Public API surface (v1.2)

The public header is `include/lc/lc.h`. Pure C99. No template, no C++
class. All C++ complexity stays in `src/lc.cpp` (impl). The full
function-by-function spec is in [`ABI.md`](../ABI.md) at the repo root.

```c
#include "lc/lc.h"

typedef struct lc_tissue lc_tissue_t;

/* Lifecycle */
lc_tissue_t* lc_create_1d(size_t n_cells);
lc_tissue_t* lc_create_2d(size_t H, size_t W);
lc_tissue_t* lc_create_from_buffer_1d(const uint16_t* buf, size_t n);
void         lc_destroy(lc_tissue_t* t);

/* Configure */
void lc_set_threads(lc_tissue_t* t, int n);
void lc_set_pulse_value(lc_tissue_t* t, uint16_t v);
void lc_set_kernel(lc_tissue_t* t, int kernel_id);    /* ABI 1.2 */
void lc_set_sig_delta(lc_tissue_t* t, int sd);        /* ABI 1.2 */

/* Inject */
void lc_inject_1d(lc_tissue_t* t, size_t pos, size_t len, uint16_t v);
void lc_inject_2d(lc_tissue_t* t, size_t r, size_t c, uint16_t v);

/* Step */
void lc_step(lc_tissue_t* t, int n_gens);
void lc_step_adaptive(lc_tissue_t* t, int n_gens);     /* 2D only */
int  lc_step_until_stable(lc_tissue_t* t, int max_gens); /* 1D only */

/* Observe */
size_t lc_active_count(const lc_tissue_t* t);
size_t lc_r_eff(const lc_tissue_t* t);
size_t lc_get_field(const lc_tissue_t* t, uint16_t* out, size_t max_n);
size_t lc_active_chunks(const lc_tissue_t* t, size_t* out, size_t max);
```

## Repository layout

```
local-coherence/
├── include/lc/lc.h           public C99 API header
├── src/                      runtime implementation (15 CORE headers + lc.cpp)
├── benchmarks/               paper experiment drivers (e15 PAD, e16 stabilization, …)
├── benchmarks/silero_protocol/   VAD evaluation reproducing TEN-VAD's official protocol
├── apps/                     example applications on public datasets (NAB, VAD, KWS, HAI)
├── tests/                    19 unit tests / 63 assertions (test_core.cpp)
├── examples/                 5 small demos (1D minimal, freeze speedup, anomaly, …)
├── paper/                    preprint.md + 5 figures
├── docs/                     architecture, paradigm, applications, sprints, milestones
├── tools/                    build, regression, evaluation utilities
└── lab/                      golden numbers + walls-audit notes (technical reference)
```

## What the runtime guarantees

1. **Determinism**: under the same kernel, `sig_delta`, and pulse, the
   field after `N` steps is byte-identical across runs.
2. **Cross-architecture bit-exact**: same byte-identical field on
   AMD Zen 2 / Zen 4, Windows MinGW / Linux gcc 13.3. Verified by the
   regression suite for every `EXACT` golden number.
3. **Bandwidth-bound performance**: 10–12 GB/s sustained on the
   reference hardware, regardless of working set (4 KB to 64 MB), due
   to hardware-prefetcher absorption of the cache hierarchy.
4. **Cost compression on sparse perturbations**: dirty + freeze yields
   520× median (≥600× peak) speedup over the naive dense kernel on a
   1 M-cell tissue × 4000 generations, with bit-exact L1-distance
   preservation.

## Build dependencies

External: none. Only the C/C++ standard library, plus `advapi32` on
Windows for `VirtualAlloc` / `VirtualLock` (large pages). The reference
build uses GCC 16.1.0 inside w64devkit on Windows; gcc 13.3 on Linux.

```bash
./tools/build.ps1          # Windows MinGW
# or
g++ -O3 -mavx2 -mfma -std=c++17 src/lc.cpp -shared -fPIC -o build/liblc.so
```

## Threading model

The runtime uses a sense-reversing barrier for multi-thread propagation
(M19) and, optionally, wave-driven workers parked on a futex/cv when
the tissue is fully frozen (M20 / Leap 11) — so power consumption drops
to near-zero in idle regimes.

A single `lc_tissue_t*` is **not** safe to operate concurrently from
multiple threads. Build separate tissues per thread (sharding), or
serialize access at the application layer.

## What is NOT in the runtime

- Bindings to scripting languages — `lcruntime-python` provides ctypes
  bindings as a separate repo
- DSL for pluggable rules — only the four built-in kernels are exposed
  in ABI 1.2
- Persistence beyond `lc_create_from_buffer` / `lc_get_field` — the
  application owns mmap, snapshot, replay
- Network or protocol parsing — application's job
- Federation, distributed coordination — application's job
