# LC Runtime — Local Coherence

**LC Runtime** is a CPU-native engine for local-propagation inference over a
continuous tissue of integer cells. **Local Coherence** is the paradigm it
implements: instead of carrying a model and running it over every input, the
runtime treats memory as a substrate where each entity (sensor, IP, frame,
event) is a cell that evolves under a fixed local rule. Stable regions are
*frozen* at zero cost; only the active front does work. Classification,
anomaly detection, voice-activity detection — they emerge from the dynamics of
the tissue, not from a trained network.

Concretely: a C99 library (~5 MB compiled), uint16 fixed-point, deterministic
and **bit-exact across architectures** (Zen 2 Windows ↔ Zen 4 Linux, byte for
byte). No GPU, no PyTorch, no ONNX. Validated as a paradigm — not a polished
product. Where it fits, it is 30–200× faster than ML baselines on the same
CPU; where it doesn't, we say so honestly ([`docs/applications.md`](docs/applications.md)).

## The principle

Under a strictly local kernel with finite spread, the per-step cost is
proportional to the **active** region — not the whole field. We call this
the **Differential Activity Principle (PAD)**. The companion paper formalizes
and validates it across 18 reproducible experiments on a 15-watt mobile CPU.

[Read the paper →](paper/preprint.md)

## What you get

- **Bit-exact** between Zen 2 Windows (MinGW) and Zen 4 Linux (gcc 13.3) — same scores, same field values, byte-for-byte
- **520×** median speedup of dirty + freeze vs naive dense kernel on sparse perturbations
- **10–12 GB/s** sustained throughput regardless of working set (4 KB to 64 MB) — bandwidth-bound, not cache-bound
- **5 MB** compiled, single C99 header API, no PyTorch / no ONNX
- Validated paradigm — not a polished product

## 30 seconds in code

```c
#include "lc/lc.h"
#include <stdio.h>

int main(void) {
    lc_tissue_t* t = lc_create_1d(1024);
    if (!t) return 1;

    lc_set_kernel(t, LC_KERNEL_CANONICAL);
    lc_set_sig_delta(t, 4);

    lc_inject_1d(t, 512, 1, 30000);
    lc_step(t, 100);

    printf("active cells: %zu  r_eff: %zu\n",
           lc_active_count(t), lc_r_eff(t));

    lc_destroy(t);
    return 0;
}
```

Build (Linux):
```sh
gcc -std=c99 -O2 -I include -L build -o hello hello.c -llc
```

Python users: see the separate
[`lcruntime-python`](https://github.com/bracoTuxbr/lcruntime-python) repo
for `pip install lcruntime` and idiomatic Python wrapper.

## Navigate

- [`paper/preprint.md`](paper/preprint.md) — the formal paper (~8000 words, 5 figures, 18 reproducible experiments)
- [`docs/paradigm.md`](docs/paradigm.md) — paradigm explained in 4 layers (5 words / 1 sentence / 1 paragraph / technical)
- [`docs/applications.md`](docs/applications.md) — where LC wins, where it loses, **plus untested ideas** worth exploring
- [`docs/architecture.md`](docs/architecture.md) — runtime / application boundary, public C API, repository layout
- [`docs/embedding-guide.md`](docs/embedding-guide.md) — how to embed `liblc` in an application (patterns, anti-patterns, diagnostics)
- [`ABI.md`](ABI.md) — normative C99 API spec, function by function (pre/post/thread-safety)
- [`benchmarks/`](benchmarks/) — driver code for paper experiments (e15 PAD, e16 stabilization, e17 roofline, e19 multi-thread)
- [`benchmarks/silero_protocol/`](benchmarks/silero_protocol/) — VAD bench reproducing TEN-VAD's official protocol
- [`examples/`](examples/) — short demos (1D minimal C++ + 3 C samples (pulse, anomaly, mel))
- [`tests/test_core.cpp`](tests/test_core.cpp) — 19 unit tests / 64 assertions
- [`AUTHORS.md`](AUTHORS.md) — human + AI collaboration provenance
- [`CHANGELOG.md`](CHANGELOG.md) — release history through ABI 1.2

Python users — see the separate
[`bracoTuxbr/lcruntime-python`](https://github.com/bracoTuxbr/lcruntime-python)
repo for the pip-installable wrapper and Python examples (NAB anomaly,
VAD, KWS, HAI detectors).

## What we tried, honestly

| Domain | Result | Verdict |
|---|---|---|
| Time-series anomaly (NAB iter4) | LC wins per-file F1 in 4/4 baselines | **Win** |
| Audio binary classification (M9 LOO) | 82.9% no NN, no GPU | **Win** |
| VAD music+speech mix (Reels) | LC ROC-AUC 0.86 vs Silero 0.79 | **Niche win** |
| DDoS detection POC | Sub-second + FP<1% on synthetic | **In trial** |
| VAD clean speech (TEN testset) | LC AP 0.91 vs TEN/Silero 0.985 | Honest loss |
| KWS Google Speech Commands | MFCC beats LC (0.45 vs 0.21 acc) | Honest loss |
| Anti-spam IP ranking | Python Counter beats LC | Honest loss |
| HAI industrial control | Mahalanobis beats LC | Honest loss |

**LC is not a universal champion.** It is a paradigm with specific sweet spots
where locality, sparsity, and freeze pay off. Where they don't, LC loses
honestly to specialized methods.

See [`docs/applications.md`](docs/applications.md) for full per-domain analysis.

## Where LC is the right tool

- Workloads where input is locally structured (time-series, network telemetry, audio frames)
- Sparse-perturbation regimes where most cells are stable (freeze pays off)
- Deployment targets without GPU / where determinism cross-arch matters
- Edge inference at sub-watt per task

## Where LC is the wrong tool

- Dense matrix multiplication / global-attention transformers
- Workloads where most cells change every step
- Fine-grained multi-class classification (KWS, semantic discrimination)
- Anywhere a well-trained ML model already dominates

## Build

```bash
git clone https://github.com/bracoTuxbr/local-coherence
cd local-coherence
./tools/build.ps1
```

Tested on Windows MinGW64 + Linux gcc 13.3. C++17 required.

## Python binding

`pip install lcruntime` — separate repo at
[`bracoTuxbr/lcruntime-python`](https://github.com/bracoTuxbr/lcruntime-python).

## Provenance

This project was built in **collaboration between Thiago Alencar (human) and
Anthropic Claude (AI assistant)**, over **6 days** of intensive sessions
(2026-05-07 to 2026-05-12). Claude wrote most of the code and the paper text;
Thiago set direction, ran the trials in real infrastructure, and made every
strategic call.

We make this explicit because (i) it is honest, and (ii) it changes how the
work was produced. The empirical claims — bit-exact golden numbers, benchmarks,
cross-architecture validation — stand on their own and can be reproduced
without trusting either author.

See [AUTHORS.md](AUTHORS.md) for the full provenance statement.

## Citing

```bibtex
@misc{alencar2026localcoherence,
  title  = {Local Coherence: empirical validation of the Differential
            Activity Principle on a 15-watt mobile CPU},
  author = {Alencar, Thiago},
  note   = {Implementation and experiments developed with Anthropic Claude
            (Opus 4.6 / 4.7); see AUTHORS.md for full provenance.},
  year   = {2026},
  url    = {https://github.com/bracoTuxbr/local-coherence}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
