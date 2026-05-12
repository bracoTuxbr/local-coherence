# Embedding guide

Practical guide for using the `liblc` C runtime in an application.
For the normative function-by-function spec see [`../ABI.md`](../ABI.md);
for the architectural boundary between runtime and application see
[`architecture.md`](architecture.md).

## Hello World

```c
#include "lc/lc.h"
#include <stdio.h>

int main(void) {
    lc_tissue_t* t = lc_create_1d(1024 * 1024);
    if (!t) return 1;

    /* inject pulse at the center */
    lc_inject_1d(t, 524288, 1, 60000);

    /* propagate 50 generations */
    lc_step(t, 50);

    /* observe */
    printf("Active cells: %zu\n", lc_active_count(t));
    printf("Effective radius: %zu\n", lc_r_eff(t));

    lc_destroy(t);
    return 0;
}
```

Build:
```sh
gcc -std=c99 -O2 -I include -L build -o hello hello.c -llc
```

Run (Windows: `liblc.dll` must be in PATH or the current directory):
```sh
./hello
```

## Common patterns

### 1. Anomaly detection on a time series

```c
lc_tissue_t* t = lc_create_1d(8192);

/* ingest data in a sliding window */
for (int i = 0; i < n_samples; ++i) {
    size_t pos = i % 8192;
    lc_inject_1d(t, pos, 1, sample_value[i]);
    lc_step(t, 1);

    size_t a = lc_active_count(t);
    if (a > baseline_active * 1.5) {
        /* anomaly: spike in active region */
        emit_alert(i);
    }
}
```

Perfect match for: network flows/sec, log rate, environmental sensor.

### 2. Audio pre-processing

```c
/* mel-spec [80 x n_frames] -> 2D tissue */
lc_tissue_t* t = lc_create_2d(80, n_frames);
for (int m = 0; m < 80; ++m)
    for (int f = 0; f < n_frames; ++f)
        lc_inject_2d(t, m, f, mel[m * n_frames + f]);

/* propagate adaptive */
lc_step_adaptive(t, 100);

/* extract active regions (= regions with signal) */
size_t dirty[1024];
size_t n = lc_active_chunks(t, dirty, 1024);
/* feed Whisper only with frames covered by dirty tiles */
```

### 3. DDoS detection

```c
/* 1D tissue = aggregation by src_ip hash */
lc_tissue_t* t = lc_create_1d(65536);
lc_set_pulse_value(t, 30000);

/* packet loop */
while (read_packet(&pkt)) {
    size_t pos = hash(pkt.src_ip) % 65536;
    lc_inject_1d(t, pos, 1, 30000);
    lc_step(t, 1);

    if (lc_active_count(t) > threshold) {
        trigger_bgp_announce();
    }
}
```

## Anti-patterns (do not do)

### Operating the same tissue from multiple threads

```c
/* THREAD A */ lc_step(t, 10);
/* THREAD B */ lc_inject_1d(t, ...);  /* RACE: undefined behavior */
```

Solution: external lock, OR use `lc_set_threads(t, N)` for internal parallelism
(in v1.1+ once MT is implemented).

### Using a tissue that is too small

```c
lc_tissue_t* t = lc_create_2d(64, 64);   /* only 4 KB */
```

For tissues that fit in L2/L1 (~256 KB), a traditional float32 stencil
implementation (Eigen, scipy.ndimage) is ~2.5× faster than LC because the
bitmap overhead does not pay off. **Use LC only for tissues > 1M cells.**

### Expecting automatic long-range correlation

```c
lc_inject_1d(t, 0, 1, 60000);
lc_inject_1d(t, 1000, 1, 60000);
lc_step(t, 100);
/* Expecting the tissue to "link" the two points? NO. */
```

Propagation has an effective horizon of ~32 cells (pulse 60000). For
long-range correlation, use:
- Continuous source repaint (re-inject every step)
- Global post-propagation decoder (extract features from global
  active_count)
- 2-level hierarchy (planned for v2)

See paper §5.2 for the empirical-horizon characterisation.

### Mixing application logic into the runtime

```c
/* do not do this */
if (parse_dns(pkt) == TUNNELING) {
    lc_inject_1d(t, ...);
}
```

The runtime does not know DNS, audio, network. The application does
parsing + heuristics; the runtime only propagates and measures activation.
Clear separation between layers.

## Expected performance

Reference table (Ryzen 5 7520U Zen 2, single-threaded in v1.0):

| operation | n=1M | n=4M | n=16M |
|---|---:|---:|---:|
| `lc_step` (sparse) | ~0.7 µs/gen | ~3 µs/gen | ~10 µs/gen |
| `lc_step` (uniform) | ~400 µs/gen | ~2.3 ms/gen | ~9 ms/gen |
| `lc_step_adaptive` 2D 2048² (sparse) | — | ~770 µs/gen | — |
| `lc_inject_1d` | <1 µs | <1 µs | <1 µs |
| `lc_active_count` | ~100 µs | ~400 µs | ~1.5 ms |
| `lc_active_chunks` (256 dirty) | ~1 µs | ~2 µs | ~3 µs |

Use `lc_active_chunks` instead of `lc_active_count` for frequent monitoring
— it is O(n_chunks/64) vs O(n_cells).

## Diagnostics

```c
int maj, min;
lc_abi_version(&maj, &min);
if (maj != LC_ABI_MAJOR) {
    fprintf(stderr, "ABI mismatch: header=%d.%d, library=%d.%d\n",
            LC_ABI_MAJOR, LC_ABI_MINOR, maj, min);
    return 1;
}
fprintf(stderr, "Build: %s\n", lc_build_info());
```

## Current limitations (ABI 1.2)

- **Single-threaded internally**: `lc_set_threads(t, >1)` is recorded but
  not honored — propagation runs on one thread regardless. Multi-thread
  use today: shard a workload across multiple tissues (one per thread).
- **`lc_step_adaptive` is 2D-only**. For 1D adaptive, the adaptive
  helper exists internally but is not exposed in the public API.
- **`lc_step_until_stable` is 1D-only**.
- **Long-range correlation**: under the canonical kernel, ~32 cells
  before the front decays below `uint16` representable (M24).
- **No built-in persistence**: use `lc_get_field` + your own file I/O,
  or `lc_create_from_buffer_1d` to resume from a saved buffer.

## What is already in (and what isn't)

| feature | status |
|---|---|
| C99 public API, opaque tissue handle | shipped in v1.0 (2026-05-08) |
| Python ctypes wrapper | shipped in v1.1 in [`lcruntime-python`](https://github.com/bracoTuxbr/lcruntime-python) (2026-05-09) |
| `lc_set_sig_delta` runtime tunable | shipped in v1.1 |
| `lc_set_kernel` (3 pluggable kernels on the public uint16 API) | shipped in v1.2 (2026-05-10) |
| `HotField<T>` for `uint8 / uint16 / uint32` | shipped in v1.2 |
| Internally-honored multi-thread propagation | not yet |
| Built-in mmap persistence | not yet |
| 2-level tissue hierarchy | not yet |
| Rule via callback / DSL | not yet |

## Reporting bugs

`lc_build_info()` returns a build identifier. Always include it in bug
reports together with:
- Operating system + architecture
- Tissue dimensions
- Call sequence that reproduces the problem
- Expected vs observed output
