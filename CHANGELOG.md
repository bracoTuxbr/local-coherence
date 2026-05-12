# Changelog

All notable changes to this project. Dates in ISO 8601.

## Unreleased — 2026-05-12

Public-divulgation polish: this is the curated cut of the research work for
sharing beyond the original author.

### Added

- **AUTHORS.md** — explicit AI-collaboration provenance statement
  (Thiago Alencar + Anthropic Claude over 6 days)
- **docs/paradigm.md** — four-layer explanation (5 words / 1 sentence / 1 paragraph / technical)
- **docs/applications.md** — 4 wins + 4 losses + untested ideas where LC might fit
- **docs/architecture.md** — runtime vs application boundary, public C API, repository layout
- **docs/embedding-guide.md** — practical guide to embedding `liblc` in an application
- **benchmarks/silero_protocol/** — VAD bench reproducing TEN-VAD's
  official PR-curve protocol on their public 30-wav testset, plus our
  separate ESC-50 + Instagram Reels evaluation
- **AI-assistance disclosure** in paper §7.3, README, CITATION.cff

### Changed

- Paper preprint sanitized of production-environment details (specific
  cluster size, vendor names, internal hostnames). Empirical content
  unchanged; descriptions generalized.
- Repository fully translated from Portuguese to English. The only
  intentionally preserved Portuguese string is *"Coerência Local"* in
  AUTHORS.md (the official PT name of the paradigm).

### Removed

- Apps that processed private data (`anti_spam`, `anti_ddos`,
  `anti_brute_force`) — their parsers and detectors live in a private
  research fork. Mentioned for completeness in `docs/applications.md`.
- Internal-only documentation: defensive doc for agents modifying
  CORE files, v1.0 scope-planning doc, internal sprint listing,
  milestone narrative timeline, wall-audit drivers. All of these
  guided the research but do not inform a public reader.
- Daily research journals and 24-file per-milestone raw markdown
  history.
- Production-environment references throughout the codebase (specific
  cluster names, vendor names, IP ranges, internal hostnames).

## 2026-05-10 — ABI 1.2

### Added

- **`lc_set_kernel(lc_tissue_t*, int kernel_id)`** in the public C API:
  pluggable propagation rule. Four kernels available:
  - `LC_KERNEL_CANONICAL` (default, bit-exact baseline)
  - `LC_KERNEL_SIMPLE_AVG`
  - `LC_KERNEL_EMA` (faster decay)
  - `LC_KERNEL_CANONICAL_U64` (for hash-collision counters)
- Underlying `HotField<T>` generalization: cell type can now be
  `uint8_t`, `uint16_t`, or `uint32_t`. The default tissue remains
  `HotField16` (uint16) for backward compatibility with every v1.0
  EXACT golden number.
- Python wrapper updated to expose `set_kernel(t, KERNEL_ID)` in
  the separate [`lcruntime-python`](https://github.com/bracoTuxbr/lcruntime-python)
  repo.

### Validations

- W1 audit cleared: kernel templated as `HotField<T>`; pulse 1 M cells
  × 4000 generations on three types, cross-architecture bit-exact
  verified.
- W2 audit cleared: kernel as template parameter via
  `propagate_chunk_t<T, Kernel>`; four kernels validated cross-arch.

## 2026-05-09 — ABI 1.1

### Added

- **`lc_set_sig_delta(lc_tissue_t*, int sd)`** in the public C API:
  runtime-tunable significance threshold for the dirty bitmap. Default
  remains `sd=4` (preserves all v1.0 EXACT golden numbers). `sd=1`
  makes the dirty path bit-exact with a naive (no-freeze) reference.

### Validations

- W3 audit: multi-workload validation of `sig_delta` runtime control.
  **Discovery**: `sig_delta=1 → L1=0` (dirty matches naive byte-for-byte)
  on every test workload (pulse, ramp, burst, adversarial uniform), on
  both laptop and server hardware. This makes the dirty path a strict
  superset of the naive baseline and lets users trade L1 accuracy for
  speedup at will.

## 2026-05-08 — research milestones around v1.0 (compressed)

This day produced an unusual amount of empirically-validated work. In
chronological order, the major commits:

- **Cross-arch reproducibility infra** (`lab/lc_evolve/`) and
  cross-arch sweep Day 1–2: cross-architecture validation between laptop
  (Zen 2 Windows) and VPS (Zen 4 EPYC Linux), `CHUNK_CELLS` sweep.
- **Day 3 audit**: discovery that `L1=94148` is specific to
  `CHUNK_CELLS=64`, not a universal invariant. Cross-arch identity
  holds for every value tested.
- **Day 4 audit**: 13 candidate walls documented, with empirical
  verdicts (refuted / confirmed / documented as design).
- **NAB anomaly detection**, iterated:
  - First shot (honest): F1 = 0.211 (paradigm OK, scoring weak)
  - Iter 2: LC beats Skyline z-score in fair comparison (F1 0.287 vs 0.209)
  - Iter 3: global threshold + 4 baselines fair — LC wins all 4
  - Iter 4: auto-tune + multi-tenant — per-file F1 ceiling 0.315
- **VAD on Instagram Reels** (M8-VAD): first real victory vs incumbents,
  F1 0.946 vs WebRTC 0.898.
- **KWS Google Speech Commands**: LC features lose to plain MFCC
  (0.213 vs 0.452 accuracy) — multi-class fine-grained discrimination
  is outside LC's representational capacity.
- **Anti-spam discovery**: a plain Python `Counter` beats LC at
  aggregate IP ranking (F1 0.69 vs 0.58). LC is overkill for dense
  aggregation.
- **Paper §5.8 real-world validation**: cross-arch + `sig_delta=1` over
  an anonymized sample of production logs.
- **Sprint v1.0** finalized below.

## v1.0.0 — 2026-05-08 (alpha)

First release with an embeddable public C API.

### Added — runtime

- **Public C99 API** in `include/lc/lc.h` (16 functions, ABI 1.0)
- **`liblc.dll` / `liblc.so`** shared library, GCC + Clang buildable
- **3 standalone C demos**: `pulse_1d.c`, `anomaly_1d.c`, `mel_2d.c`
- **Python ctypes wrapper** (ABI 1.1) — distributed as a separate
  package: [`bracoTuxbr/lcruntime-python`](https://github.com/bracoTuxbr/lcruntime-python)
- **`HotField<T>` template** — `uint8 / uint16 / uint32` cell types
- **Pluggable kernel** at runtime: `CanonicalKernel`, `SimpleAvgKernel`,
  `EmaKernel`, `CanonicalKernel_u64`; selectable via `lc_set_kernel(t, ID)`
  (ABI 1.2)
- **Bit-exact regression preserved** across the v1.0 refactor

### Scientific validations (Sprint v0)

- **#27** Strict-aliasing UB in CPUID fixed (`uint32_t` reads via memcpy)
- **#28** `alloc_tissue` with size_t overflow check
- **#51** SoA vs AoS empirically defended: SoA beats AoS by 4.8–8.0× in
  DRAM (M21)
- **#16** PAD Conjecture 1 validated: log-log slope = 0.89, R² = 0.99 (M22)
- **#53** Freeze breakdown regime characterized: uniform = 0.62× speedup
  (M23). Adaptive M5.5 covers this via mode switching.
- **#17** PAD long-range limit found: k_max ≈ 32 cells (M24). Abrupt
  cliff at k = 48.
- **#59** M5.5 adaptive detector robust on real mel: 10/10 Instagram
  samples, 1 transition each (M25)
- **#56** LC vs float32 baseline (Eigen-equivalent): in DRAM sparse LC
  wins 3.90×, in DRAM uniform LC-naive wins 1.24×, in L2-fit baseline
  wins 2.5× (M26)

### New golden numbers

```
e21|aos_vs_soa_ratio_16K|4.82|20|PERF
e21|aos_vs_soa_ratio_4M|6.81|20|PERF
e22|pad_slope|0.89|10|PERF
e22|pad_r2|0.99|2|PERF
e23|adversarial_uniform_amp8_speedup|0.62|30|PERF
e23|adversarial_uniform_amp32_speedup|0.74|30|PERF
e24|long_range_kmax_v60000|32|0|EXACT
e24|long_range_gen_merged_k16|9|0|EXACT
e25|adaptive_mel_real_max_transitions|1|0|EXACT
e25|adaptive_mel_real_robust_pct|100|0|EXACT
e26|lc_adaptive_vs_float_sparse_dram|3.90|25|PERF
e26|lc_naive_vs_float_uniform_dram|1.24|25|PERF
```

### Structural refactor

- `src/` — 15 CORE paradigm headers (flattened in the public repo from
  the nested layout used during research)
- `benchmarks/` — pure-runtime experiment drivers
- `tools/` — build, regression, utilities

The audio and synthetic-DDoS frontends used during research live in the
private research repo and are not part of the public runtime.

### Known limitations in v1.0

- Single-threaded internally (multi-thread in v1.1)
- `lc_step_adaptive` is 2D-only
- `lc_step_until_stable` is 1D-only
- Long-range correlation limited to ~32 cells under canonical kernel
- No built-in persistence (use `lc_get_field` + file I/O)

## Pre-release history (2026-05-07)

The paradigm was validated incrementally across milestones M0–M10 plus
several follow-up "M-prime" experiments. Full per-milestone results are
summarized below. Highlights:

### M-2 — Differential Activity Principle (PAD) validated bit-exact

- 1D: `|A(t)| = 2t + 1` for `t = 0..7` (Manhattan diamond)
- 2D 5-point: `|A(t)| = 2t² + 2t + 1` for `t = 0..4`
- Effective horizon `r* ≈ log(v₀)/log(D)` confirmed empirically

### M-3 / M-4 — Stabilization

- Convergence bit-exact to zero in 510 generations
- Peak active radius `r_eff = 23` cells

### M0 — Linear-scan baseline

Confirmed the canonical kernel is bandwidth-bound, not compute-bound.

### M2.5 — Trapezoidal tile blocking in time

Process K generations entirely in L1.

### M4 — Dirty bitmap + freeze

**520× median speedup** (≥600× peak) over a naive dense kernel on sparse
perturbations, with bit-exact L1-distance preservation. This is the
defining performance result.

### M5 — 2D propagation

`src/propagate_dirty2d.hpp`, used by the audio pipeline downstream.

### M5.5 — Adaptive runtime with hysteresis

Hysteretic switching between sparse and dense regimes; avoids thrashing
when the active set fluctuates around the threshold.

### M6 — Entropy distinguishes structure from noise

Silence: 0.7% active. White noise: 83% active.

### M7 — Roofline analysis

Peak L1 12.32 GB/s, DRAM 9.93 GB/s, peak/DRAM 1.24×. Hardware prefetchers
absorb the cache hierarchy in linear access.

### M8 — VAD pre-filter for whisper.cpp

End-to-end validation on real Instagram audio. The user's original
use case.

### M9 — Emergent classification

**82.9% binary accuracy** on speech vs noise via mel → 2D LC tissue →
stability features → leave-one-out classifier. No neural network, no
GPU, no training of LC parameters.

### M10 — Anomaly detection + DDoS POC

Per-file F1 wins across all 4 baselines on NAB (Mahalanobis, EWMA,
Isolation Forest, Z-score). Foundation for the production DDoS trial.

### M19 — Multi-thread + dirty + freeze

`benchmarks/e19.cpp`. Bit-exact across thread counts 1..N via the
sense-reversing barrier.

### M20 — Wave-driven workers (zero CPU when idle)

`benchmarks/e20.cpp`. Workers sleep on futex when there is no
perturbation. Power consumption drops to near-zero in stable regime.

### M21–M24 — Runtime engineering for v1.0

ABI 1.0 → 1.1 → 1.2 progression. Templated `HotField<T>`, pluggable
kernel parameter.

### M25 — H4 dossier: adaptive detector on real mel

Confirmed robustness of M5.5 outside the synthetic regime.

### M26–M28 — Production-environment trials (private fork)

The runtime was wired into three production-style POCs during these
milestones — among them NetFlow-based DDoS detection (sub-second + FP <1%
on synthetic), plus auth/log analysis pipelines. Implementations handle
vendor-specific log formats and live in a private fork; only the latency
and false-positive constraints they exposed propagated back into the
public benchmarks.

### Infrastructure (Sprint v0)

- Protection system: `tools/regression_test.ps1`, `benchmarks/golden_numbers.txt`,
  unit tests gating bit-exact preservation of every `EXACT` invariant
- 19 unit tests / 64 assertions in `tests/test_core.cpp`
- Security review baseline + WAV-loader hardening
- Hardware-portable PERF tolerances (15–30% drift documented)

### Paper preprint v0.x

~8000-word preprint with 5 figures (PAD 1D, PAD 2D, stabilization,
roofline, CPU%-vs-time), Appendix A (golden numbers), Appendix B
(hardware), 12 references. See [`paper/preprint.md`](paper/preprint.md).
