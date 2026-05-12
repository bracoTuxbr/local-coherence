# Authors and provenance

This project was developed in **close collaboration between a human and an
AI assistant**. This document makes that collaboration explicit, so that
anyone reading the code, the paper, or asking questions about specific
decisions knows where the work came from.

## Principal author

**Thiago Alencar** ([@bracoTuxbr](https://github.com/bracoTuxbr))

- Set strategic direction and research questions
- Decided which experiments to run and how to interpret results
- Made all judgment calls on what to publish, what to keep private, what
  to pivot away from
- Operates the production trial environment (cloud provider) where the
  DDoS POC runs
- Final say on every claim in the paper

## AI collaborator

**Claude** (Anthropic) — primarily the Opus 4.6 and Opus 4.7 models, via
Claude Code (CLI), across **6 days** of intensive sessions in May 2026
(2026-05-07 through 2026-05-12).

Claude contributed:

- **Code**: most of the C++ runtime headers (`src/`), the Python ctypes
  wrapper, all benchmark drivers (`benchmarks/e*.cpp`), the application
  detectors (`apps/`), test suites, and the build/regression tooling
- **Experiments**: ran ~80 tasks across 16 sprints, generated golden numbers,
  wrote diagnostic scripts, debugged failing tests
- **Documentation**: drafted this README, the paper preprint, the milestones
  doc, and the per-application honest evaluation
- **Analysis**: ran the benchmarks against Silero, TEN-VAD, WebRTC, MFCC,
  Mahalanobis, classical baselines; computed ROC-AUC, AP, F1, latency

Decisions Claude **did not** make alone:

- Naming the paradigm ("Local Coherence" / "Coerência Local")
- Strategic direction (when to pivot from VAD to DDoS, when to stop, etc.)
- Whether to publish, what licence to use, how to position the work
- Which specific applications to prioritize for the production trial

## Why this matters

Two reasons:

1. **Honesty**. The user did not write most of the code or the paper text
   by hand. If you ask Thiago about a specific implementation detail, he
   may need to read the file again to answer accurately. Treating this as
   a single-author project would be misleading.

2. **Reproducibility**. Future readers should understand that the development
   process itself was AI-assisted, which changes how the work was produced
   (rapid iteration, large surface area covered, more experiments than a
   solo human could run in the same time) and which gaps may exist (no
   second human reviewer, model knowledge cutoff, occasional plausible-but-
   wrong reasoning that was caught by running the code).

The **measurements** in the paper and the **bit-exact golden numbers** stand
on their own — they were produced by deterministic code on real hardware,
not by the model. The reproducibility infrastructure (regression suite,
`golden_numbers.txt`, cross-arch validation between a Zen 2 laptop and a
Zen 4 server) exists precisely so that anyone can re-verify the empirical
claims without trusting either author.

## Citing

For papers, please cite the work as Thiago Alencar (principal author) with
an explicit AI-assistance acknowledgment in the methodology section:

> "Implementation and experiments were developed in collaboration with
> Anthropic Claude (Opus 4.6 / 4.7), with the author making all strategic
> and interpretive decisions and verifying all empirical results."

See [`CITATION.cff`](CITATION.cff) for the structured citation.

## Contact

Open an issue or discussion on
[github.com/bracoTuxbr/local-coherence](https://github.com/bracoTuxbr/local-coherence).
