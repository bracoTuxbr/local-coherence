# The paradigm in four layers

Different audiences need different depths. Pick yours.

## Layer 1 — five words

> Stability-driven inference on raw CPU.

## Layer 2 — one sentence

A CPU-native runtime where inference emerges from the stabilization of a tissue
of cells in continuous memory — no central training, no GPU.

## Layer 3 — one paragraph

Most inference today carries a giant model and runs it over each input via GPU.
Local Coherence treats memory as a physical continuous substrate where each
entity (user, flow, sensor, account) is a cell that evolves in time under local
rules. Stable patterns become *frozen* (zero cost); changing patterns propagate
waves that reveal structure. Classification emerges from the dynamics of the
tissue, not from a trained model. On a commodity CPU, this runs in nanoseconds
per event — orders of magnitude faster than traditional ML for locally-structured
streaming problems.

## Layer 4 — technical (five properties)

1. **Continuous-memory tissue** aligned to cache lines (64-byte chunks)
2. **Local waves** (immediate neighbors only) — no global passes
3. **Stabilization as compression of work** (freeze + reactivation)
4. **Adaptive switching** between sparse and dense regimes (with hysteresis)
5. **Semantic emergence** — the freeze pattern itself encodes information

## The Differential Activity Principle (PAD)

> Under a strictly local kernel with finite spread, the per-step cost is
> `Θ(|A(t)|)`, where `A(t)` is the set of cells whose value differs from
> their previous value. Cells outside the halo of the perturbation are
> *never touched*.

This is what makes LC bandwidth-efficient instead of compute-efficient:
the runtime scans a dirty bitmap and only touches active cells. Stable
regions are skipped entirely.

## Why CPU and not GPU

CPU has a memory hierarchy: register 0.3ns / L1 ~1ns / DRAM ~80–120ns.
The L1↔DRAM gap is 100×. Most software ignores this — load model into DRAM,
access randomly.

Local Coherence respects the hierarchy: 99% of work in L1 = 100× faster than
code that abuses DRAM. To make that happen, the algorithm must be local *and*
the tissue must fit in L1. Tile-blocking in time (M2.5) processes K generations
entirely in L1; freeze (M4) eliminates re-processing of stable regions;
adaptive switching (M5.5) adjusts to the regime.

**This isn't optimization — it's a design principle.** Computation that
respects the physics of the hardware, instead of assuming an abstract
uniform machine.

## Comparison with adjacent paradigms

| paradigm | how it differs |
|---|---|
| Transformer (LLM) | Global attention + training + GPU vs our local rules + no training + CPU |
| Tree models (XGBoost) | Static tabular features vs temporal streaming |
| Classical CA (Wolfram) | Symbolic deterministic computation vs statistical inference + freeze + adaptive |
| Neural Cellular Automata | Gradient-descent-learned rules + GPU vs hand-engineered emergent rules + CPU |
| Pochoir / stencil compilers | PDE/physics vs ambiguous-streaming inference + freeze + adaptive |
| Spiking NN (Loihi) | Custom neuromorphic hardware vs pure software on commodity CPU |
| Active-set methods | Optimization vs detection/classification |

Local Coherence is a synthesis of several of these ideas applied to a new space
(ambiguous-streaming inference on commodity CPU).

## What "Local Coherence" means

- **Local** because computation only touches cells in the immediate
  neighborhood of an activity front
- **Coherence** because patterns that emerge are the *stability structure*
  of the tissue — coherent regions vs incoherent ones — not features of a
  trained model

Sigla: LC (or LCR for "Local Coherence Runtime").

## Where the name comes from

After exploring "Stability-Driven Inference", "Cellular Substrate Inference",
"Topology-Native Computing", "Reactive Tissue Computing", we settled on
**Local Coherence** because:

- "Coherence" carries technical weight (physics, signal processing)
- "Local" describes alignment with the physical topology of the CPU
- Easy to pronounce, easy to google, easy to remember
