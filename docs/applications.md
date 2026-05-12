# Applications — what LC is good at, plus ideas where it might help

This document has three parts:

1. **Validated wins** — domains where LC was tested and outperformed baselines
2. **Potential applications** — domains where LC's properties suggest a fit (untested)
3. **Honest losses** — domains where LC was tested and lost (so you don't repeat)

The goal: if you land here looking for a use case, you should leave with a
clear sense of where LC might be the right tool for **your** problem.

---

## 1. Validated wins

### Time-series anomaly detection (NAB benchmark)
- **Result**: LC beat all four classical baselines (Mahalanobis, EWMA,
  Isolation Forest, Z-score) in per-file F1 score across the NAB iteration 4
  setup.
- **Why it works**: NAB streams are sparse-perturbation regimes. Most timesteps
  are stable; anomalies are local in time. Dirty + freeze + magnitude tracking
  align with this structure.
- **Where it transfers**: any single-source streaming metric where anomalies are
  rare, local, and you need a fast detector that doesn't need training.
  Reference implementation in
  [`lcruntime-python/examples/nab_detector/`](https://github.com/bracoTuxbr/lcruntime-python/tree/main/examples/nab_detector).

### Audio binary classification (M9: speech vs noise)
- **Result**: 82.9% binary accuracy on a leave-one-out benchmark of real
  Instagram audio, **with no neural network and no GPU**.
- **Pipeline**: mel-spectrogram → 2D LC tissue → stability features → simple
  classifier.
- **Why it works**: per-frame stability of the LC field encodes structural
  information. The classifier reads only the dynamics, not the raw signal.
- **Where it transfers**: binary "is something happening" detection on any 2D
  spectrogram-like input (sound, vibration, radio spectrum).

### Voice Activity Detection on music+speech mix
- **Result**: with the calibrated `simple_avg + steps=8 + sig_delta=1` config,
  LC reaches ROC-AUC 0.859 on Instagram Reels + ESC-50 noise, vs Silero V5
  0.788 and TEN-VAD 0.803. The default (`canonical, sig_delta=4`) config
  scores 0.816, still above both ML models on this dataset. On the same
  hardware, LC is **16× faster than TEN-VAD and 57× faster than Silero**
  (13 / 209 / 748 µs per 32 ms chunk on the reference laptop).
- **Why it works**: ML VADs were trained on clean speech and confuse music
  for speech. LC's magnitude tracking is content-agnostic — speech and music
  both fire the tissue, but the temporal envelope still discriminates.
- **Where it transfers**: VAD for podcasts with music intros, TikTok/Reels
  short-form content, streaming with background music. See
  [`benchmarks/silero_protocol/`](../benchmarks/silero_protocol/) for
  the PR-curve harness; Python reference implementation in
  [`lcruntime-python/examples/vad_detector/`](https://github.com/bracoTuxbr/lcruntime-python/tree/main/examples/vad_detector).

### DDoS detection (cloud provider trial)
- **Result**: synthetic-attack benchmark — sub-second detection, FP rate <1%.
- **Status**: in trial deployment at a Brazilian cloud hosting provider.
- **Why it works**: DDoS bursts are sparse-perturbation events. LC's per-IP
  cell freezes during normal traffic and fires on anomalous activity. No model
  training required.
- **Where it transfers**: network telemetry, intrusion detection at the
  firewall level, anything where "most sources are normal most of the time"
  holds.
- *Implementation lives in a private research fork that handles
  vendor-specific log and telemetry formats. Not included here.*

### Audio activity detection (general "is something happening")
- **Use**: tracked across M8 (VAD pre-filter for whisper.cpp), M9 (binary
  classification), and the VAD bench (32 ms chunks, ~96K chunks/sec on the reference laptop).
- **Where it transfers**: podcast auto-edit (cut silences), surveillance
  any-sound detection, sports highlights (audio peaks), broadcast
  auto-leveling.

---

## 2. Potential applications (untested, but the properties match)

These are domains where LC's structural fit is strong but we have not
benchmarked them. If you work in any of these and try LC, we want to hear
how it goes.

### Network / security

- **Bot detection on web logs** — each visitor = one cell, click/request
  events propagate. Sparse anomalies (sudden burst from one IP) fire freeze
  reactivation.
- **DNS tunneling detection** — per-domain anomaly via dnstap stream. Already
  drafted as extension of DDoS work (~500 LOC, 80% reuse).
- **BGP route anomaly** — global routing telemetry is locally structured
  (most ASes are stable most of the time).
- **Click fraud / ad-tech anomaly** — per-publisher per-source cell, sparse
  fraud bursts.
- **Mail-server log anomaly** — per-IP cell on event streams. We tried
  this on aggregate IP ranking in private data and a Python `Counter`
  beat LC; the win *might* be on the temporal-burst angle instead.

### Audio / signal

- **Onset detection** for drums and percussion — envelope velocity is exactly
  what the tissue tracks.
- **Spectral envelope tracking** for auto-leveling and broadcast loudness
  normalization.
- **Beat tracking** when combined with FFT pre-processing (LC operates on
  spectral magnitude over time).
- **Cognitive radio spectrum sensing** — real-time, sub-watt detection of
  primary-user signals on shared bands.

### Industrial / IoT

- **Vehicle CAN bus anomaly** — each CAN ID = one cell, fault events propagate
  locally in the message stream.
- **Smart-home presence/activity** — sensor events fire freeze reactivation.
- **Sensor calibration drift detection** — slow drift is exactly what
  long-horizon LC dynamics surface.
- **Predictive maintenance vibration analysis** — accelerometer streams have
  the locality property (most timesteps stable, faults are sparse).

### Observability / telemetry

- **Application metrics anomaly** — each metric = one cell on a 1D tissue,
  spikes propagate. Cheaper than Prometheus alert rules at scale.
- **Log line anomaly per service** — service = cell, event rate fires
  perturbations.
- **Distributed tracing burst detection** — per-endpoint cells.

### Edge / embedded

- **Microcontroller inference** — 5 MB compiled, uint16 only, no FPU
  required. Could run on Cortex-M class hardware with adaptation.
- **Raspberry Pi at the edge** — Python binding works, sub-watt inference
  on commodity SBC.

### Suggestions to explore (just hypotheses)

- **Game AI**: NPC reactions to local events (player proximity, sound,
  damage) in a tissue keyed by spatial position.
- **Touch / haptic event clustering** — tactile sensor streams.
- **High-frequency trading order flow anomaly** — per-symbol cell, order
  events propagate, anomaly = freeze reactivation.
- **Particle effect visualization** — render-friendly because all state is
  in a contiguous uint16 buffer.

---

## 3. Honest losses

We tested LC in these domains and other approaches won. Knowing where LC
loses is as valuable as knowing where it wins.

### VAD on clean speech (TEN-VAD official testset)
- **Result**: LC canonical AP **0.91** vs TEN-VAD **0.985** vs Silero V5
  **0.985** (technical tie between the two ML models).
- **Why it loses**: in clean speech, ML models trained on librispeech and
  gigaspeech learn precise speech/non-speech features that a linear stencil
  cannot replicate.
- **Lesson**: LC is *content-agnostic*, which helps in music+speech mix and
  hurts in clean speech.

### Keyword spotting (Google Speech Commands)
- **Result**: LC features + LogReg accuracy 0.21 vs MFCC + LogReg 0.45 across
  10 keywords.
- **Why it loses**: KWS requires fine-grained spectral discrimination across
  many classes. LC's scalar magnitude per chunk does not have the
  representational capacity.
- **Lesson**: LC is poor at multi-class fine-grained classification. Use it
  as a pre-filter for ML KWS, not as a replacement.

### Mail-server log aggregate IP ranking
- **Result**: a trivial `collections.Counter` over event lines beat LC
  at F1 0.69 vs 0.58.
- **Why it loses**: aggregate IP ranking has no temporal-locality structure
  to exploit. The temporal dimension is dense (every event matters), so
  freeze has nothing to skip.
- **Lesson**: LC is overkill for dense aggregation. Use a hash table.

### HAI industrial control (multivariate anomaly)
- **Result**: LC F1 0.04 vs Mahalanobis F1 0.33 on HAI 21.03 (79 sensors,
  1-second timesteps).
- **Why it loses**: HAI anomalies are multivariate correlations (sensor 12
  should rise when sensor 47 falls). LC's 1D tissue can only encode one
  sensor at a time without explicit cross-sensor coupling.
- **Lesson**: LC is not the right tool for joint-distribution anomaly
  detection without architectural changes.

---

## When to consider LC

- ✓ Input is **locally structured** (time-series, audio frames, network
  telemetry, log streams, sensor events)
- ✓ Most cells are **stable most of the time** (sparse-perturbation regime
  — freeze pays off)
- ✓ You need **cross-architecture bit-exact determinism** (debug == prod)
- ✓ Target is **CPU-only / sub-watt edge** (no GPU available)
- ✓ Latency budget is **sub-millisecond per event**

## When to skip LC

- ✗ Dense matrix multiplication / global-attention transformers
- ✗ Most cells change every step (no freeze opportunity)
- ✗ Multi-class fine-grained classification (use MLP / transformer)
- ✗ Joint-distribution anomaly without locality (use Mahalanobis / Isolation
  Forest / autoencoder)
- ✗ Simple aggregation / counting (use a hash table)

---

## A note on "champion vs niche"

LC is **not a universal champion** in any of the mature benchmarks we tried.
Mature ML methods, with years of engineering and large training datasets, are
hard to beat in their home domain.

What LC offers is a **different operating point**: deterministic, fast,
no-training, no-GPU — that fits a class of problems where mature methods are
either unavailable, too heavy, or simply unnecessary. If your problem has the
locality + sparsity structure described above, LC is worth trying.
