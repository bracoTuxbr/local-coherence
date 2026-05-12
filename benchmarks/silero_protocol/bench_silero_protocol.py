"""bench_silero_protocol.py — VAD bench following the official Silero methodology.

Primary metric: ROC-AUC over 31.25ms chunks (500 samples @ 16kHz).
Secondary: accuracy, F1-best-threshold, us/chunk, RTS (real-time speed).

Silero protocol (ref: github.com/snakers4/silero-vad wiki):
  - chunk = 31.25ms = 500 samples
  - threshold sweep for ROC-AUC (threshold-independent)
  - 1 thread, batch=1, 16kHz mono
  - Multi-domain: combine negatives (no-speech) + positives (speech)

Datasets:
  - negatives: ESC-50 (no-speech audio — every chunk = 0)
  - positives: Reels labeled (speech audio — chunks inside segments = 1)
  - if missing: generates a silent pure-tone synthetic signal (debug)

Usage:
  python bench_silero_protocol.py [--neg-dir DIR] [--pos-dir DIR] [--max-files N]
  python bench_silero_protocol.py --neg-dir data/esc50/audio --pos-dir runs/insta_audio
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

# Requires the `lcruntime` Python binding (separate repo: bracoTuxbr/lcruntime-python).
# Install with:  pip install lcruntime
# Then point LC_LIB at the compiled liblc.so / liblc.dll built from this repo.
HERE = Path(__file__).resolve().parent

import webrtcvad

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    import torch
    _SILERO_AVAILABLE = True
except ImportError:
    _SILERO_AVAILABLE = False

try:
    from ten_vad import TenVad
    _TENVAD_AVAILABLE = True
except ImportError:
    _TENVAD_AVAILABLE = False


SAMPLE_RATE = 16000
# Silero V5/V6 native chunk = 512 samples (32ms); doc says 31.25ms but code expects 512.
# Using 512 to avoid padding artifacts in Silero internal LSTM.
CHUNK_SAMPLES = 512
CHUNK_MS = CHUNK_SAMPLES * 1000 / SAMPLE_RATE   # 32.0
# WebRTC must use 10/20/30 ms; we use 10ms internal and aggregate to 32ms (3 frames)
WEBRTC_FRAME_MS = 10
WEBRTC_FRAME_SAMPLES = SAMPLE_RATE * WEBRTC_FRAME_MS // 1000


def load_wav_mono16k(path: Path) -> np.ndarray | None:
    """Load WAV mono 16kHz. Returns float32 [-1,1] or None if incompatible."""
    try:
        with wave.open(str(path), "rb") as w:
            if w.getnchannels() != 1:
                return None
            if w.getframerate() != SAMPLE_RATE:
                return None
            sw = w.getsampwidth()
            n = w.getnframes()
            raw = w.readframes(n)
        if sw == 2:
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if sw == 1:
            return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    except Exception:
        return None
    return None


def load_wav_resample_if_needed(path: Path) -> np.ndarray | None:
    """Load WAV, resampling to 16kHz mono if necessary (via librosa)."""
    raw = load_wav_mono16k(path)
    if raw is not None:
        return raw
    try:
        import librosa
        y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        return y.astype(np.float32)
    except Exception as e:
        print(f"  skip {path.name}: {e}", file=sys.stderr)
        return None


def chunkize(audio: np.ndarray, chunk_samples: int = CHUNK_SAMPLES) -> np.ndarray:
    """Reshape audio into (n_chunks, chunk_samples). Drops trailing remainder."""
    n_chunks = len(audio) // chunk_samples
    return audio[:n_chunks * chunk_samples].reshape(n_chunks, chunk_samples)


def load_segs_to_chunk_labels(segs_path: Path, n_chunks: int) -> np.ndarray:
    """Read 10ms-frame seg labels, convert to 31.25ms-chunk labels.
    A chunk is positive if ANY 10ms frame inside it is positive.
    Conservative for speech detection — won't miss boundary speech.
    """
    chunk_labels = np.zeros(n_chunks, dtype=np.int32)
    if not segs_path.exists():
        return chunk_labels
    # 31.25ms = 3.125 * 10ms; chunk i covers frames [i*3.125, (i+1)*3.125)
    frame_ratio = CHUNK_MS / 10.0  # 3.125
    for line in segs_path.read_text().strip().split("\n"):
        parts = line.split()
        if len(parts) != 3:
            continue
        sf, ef, lbl = int(parts[0]), int(parts[1]), int(parts[2])
        if lbl == 0:
            continue
        # frames [sf, ef) → chunks
        c_start = int(sf / frame_ratio)
        c_end = min(n_chunks, int(np.ceil(ef / frame_ratio)))
        chunk_labels[c_start:c_end] = 1
    return chunk_labels


# === LC VAD (chunk-level) ===
def lc_score_chunks(audio: np.ndarray,
                    n_cells: int = 1024,
                    steps_per_chunk: int = 4,
                    sig_delta: int = 4,
                    kernel: str = "canonical") -> tuple[np.ndarray, dict]:
    """Run LC over audio at 32ms chunks. Returns score per chunk in [0,1].

    kernel: 'canonical' | 'ema' | 'simple_avg' — sweep found simple_avg+st=8 best for VAD.
    """
    import lcruntime as lc
    KMAP = {
        "canonical": lc.KERNEL_CANONICAL,
        "ema": lc.KERNEL_EMA,
        "simple_avg": lc.KERNEL_SIMPLE_AVG,
    }
    kid = KMAP[kernel]
    chunks = chunkize(audio)
    n = len(chunks)
    if n == 0:
        return np.zeros(0, dtype=np.float32), {"name": f"lc_{kernel}", "elapsed_s": 0.0}
    rms = np.sqrt(np.mean(chunks ** 2, axis=1) + 1e-12)
    rms_log = 20 * np.log10(rms + 1e-9)
    rmin, rmax = float(rms_log.min()), float(rms_log.max())
    if rmax - rmin < 1e-6:
        rmax = rmin + 1
    mag = ((rms_log - rmin) / (rmax - rmin) * 60000).astype(np.int32)
    mag = np.clip(mag, 0, 60000)
    tissue = lc.Tissue1D(n_cells)
    tissue.set_sig_delta(sig_delta)
    tissue.set_kernel(kid)
    scores = np.zeros(n, dtype=np.float32)
    t0 = time.perf_counter()
    center = n_cells // 2
    for i in range(n):
        if mag[i] > 0:
            tissue.inject(center, 1, int(mag[i]))
        tissue.step(steps_per_chunk)
        field = tissue.get_field()
        if not isinstance(field, np.ndarray):
            field = np.frombuffer(field, dtype=np.uint16)
        scores[i] = float(field.max()) / 60000.0
    elapsed = time.perf_counter() - t0
    tissue.close()
    return scores, {
        "name": f"lc_{kernel}",
        "elapsed_s": elapsed,
        "chunks_per_sec": n / max(elapsed, 1e-9),
        "us_per_chunk": (elapsed * 1e6) / max(n, 1),
    }


# === Silero ===
_SILERO_MODEL = None
def silero_score_chunks(audio: np.ndarray) -> tuple[np.ndarray, dict]:
    """Run Silero internal model directly on each 31.25ms chunk (512 samples).
    Silero V5 native chunk size is 512 samples at 16kHz; we use 500 then pad to 512.
    Returns speech probability per chunk in [0,1]."""
    global _SILERO_MODEL
    if not _SILERO_AVAILABLE:
        return np.zeros(len(audio) // CHUNK_SAMPLES, dtype=np.float32), {"name": "silero", "elapsed_s": 0.0}
    if _SILERO_MODEL is None:
        _SILERO_MODEL = load_silero_vad(onnx=False)
    chunks = chunkize(audio)
    n = len(chunks)
    scores = np.zeros(n, dtype=np.float32)
    t0 = time.perf_counter()
    _SILERO_MODEL.reset_states()
    with torch.no_grad():
        for i in range(n):
            tx = torch.from_numpy(chunks[i].astype(np.float32)).unsqueeze(0)
            prob = _SILERO_MODEL(tx, SAMPLE_RATE).item()
            scores[i] = float(prob)
    elapsed = time.perf_counter() - t0
    return scores, {
        "name": "silero_v5",
        "elapsed_s": elapsed,
        "chunks_per_sec": n / max(elapsed, 1e-9),
        "us_per_chunk": (elapsed * 1e6) / max(n, 1),
    }


# === TEN-VAD (native 256-sample frames = 16ms; aggregate 2 frames → 1 chunk) ===
def ten_vad_score_chunks(audio: np.ndarray) -> tuple[np.ndarray, dict]:
    """TEN-VAD operates at 256-sample frames (16ms). For 512-sample chunks (32ms),
    we run TEN twice and take max(prob) — preserves "any speech" semantics."""
    if not _TENVAD_AVAILABLE:
        return np.zeros(len(audio) // CHUNK_SAMPLES, dtype=np.float32), {"name": "ten_vad", "elapsed_s": 0.0}
    TEN_HOP = 256
    vad = TenVad(TEN_HOP, 0.5)
    audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    chunks = chunkize(audio)
    n = len(chunks)
    scores = np.zeros(n, dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(n):
        c = audio_int16[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
        # Two TEN frames per chunk (each 256 samples)
        p1, _ = vad.process(c[:TEN_HOP])
        p2, _ = vad.process(c[TEN_HOP:])
        scores[i] = float(max(p1, p2))
    elapsed = time.perf_counter() - t0
    return scores, {
        "name": "ten_vad",
        "elapsed_s": elapsed,
        "chunks_per_sec": n / max(elapsed, 1e-9),
        "us_per_chunk": (elapsed * 1e6) / max(n, 1),
    }


# === WebRTC (10ms internal, aggregate) ===
def webrtc_score_chunks(audio: np.ndarray, mode: int = 3) -> tuple[np.ndarray, dict]:
    vad = webrtcvad.Vad(mode)
    audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    chunks = chunkize(audio)
    n = len(chunks)
    scores = np.zeros(n, dtype=np.float32)
    # for each chunk, run WebRTC on 3 internal 10ms frames; chunk = mean(is_speech)
    t0 = time.perf_counter()
    for i in range(n):
        c = audio_int16[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
        votes = 0
        n_int = 0
        for j in range(3):  # 3 frames of 10ms inside the 31.25ms chunk (truncate)
            s = c[j * WEBRTC_FRAME_SAMPLES:(j + 1) * WEBRTC_FRAME_SAMPLES]
            if len(s) == WEBRTC_FRAME_SAMPLES:
                votes += int(vad.is_speech(s.tobytes(), SAMPLE_RATE))
                n_int += 1
        scores[i] = votes / max(n_int, 1)
    elapsed = time.perf_counter() - t0
    return scores, {
        "name": f"webrtc_m{mode}",
        "elapsed_s": elapsed,
        "chunks_per_sec": n / max(elapsed, 1e-9),
        "us_per_chunk": (elapsed * 1e6) / max(n, 1),
    }


# === Energy ===
def energy_score_chunks(audio: np.ndarray) -> tuple[np.ndarray, dict]:
    chunks = chunkize(audio)
    n = len(chunks)
    t0 = time.perf_counter()
    rms = np.sqrt(np.mean(chunks ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    # normalize to [0,1] using fixed scale (Silero-like)
    scores = np.clip((db + 60) / 60.0, 0.0, 1.0).astype(np.float32)
    elapsed = time.perf_counter() - t0
    return scores, {
        "name": "energy",
        "elapsed_s": elapsed,
        "chunks_per_sec": n / max(elapsed, 1e-9),
        "us_per_chunk": (elapsed * 1e6) / max(n, 1),
    }


# === Metrics ===
def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC via Mann-Whitney U statistic.
    AUC = P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg).
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    labels = labels.astype(np.int32)
    # Sort ascending; assign average ranks to ties
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    l_sorted = labels[order]
    n = len(s_sorted)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # ranks are 1-indexed
        ranks[i:j + 1] = avg_rank
        i = j + 1
    n_pos = int(l_sorted.sum())
    n_neg = n - n_pos
    sum_pos_rank = float(ranks[l_sorted == 1].sum())
    u = sum_pos_rank - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def threshold_sweep(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, dict]:
    if labels.sum() == 0:
        return 0.0, 0.5, {"prec": 0.0, "rec": 0.0, "tp": 0, "fp": 0, "fn": 0}
    best_f1 = 0.0
    best_thr = 0.5
    best_m = {"prec": 0.0, "rec": 0.0, "tp": 0, "fp": 0, "fn": 0}
    for thr in np.linspace(0.02, 0.98, 49):
        pred = (scores > thr).astype(np.int32)
        tp = int((pred & labels).sum())
        fp = int((pred & (1 - labels)).sum())
        fn = int(((1 - pred) & labels).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_m = {"prec": prec, "rec": rec, "tp": tp, "fp": fp, "fn": fn}
    return best_f1, best_thr, best_m


def accuracy_at_thr(scores: np.ndarray, labels: np.ndarray, thr: float = 0.5) -> float:
    pred = (scores > thr).astype(np.int32)
    return float((pred == labels).mean())


# === Pipeline ===
def collect_audio_chunks(audio_dir: Path,
                          is_positive: bool,
                          max_files: int | None = None,
                          require_segs_for_positives: bool = True
                          ) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Return list of (audio, chunk_labels, name).

    For positives: when require_segs_for_positives=True (default), only includes
    .wav files that have a matching _segs.txt (ground truth). Otherwise assumes
    all-positive for files without segs (lower-quality label).
    """
    out = []
    wavs = sorted(audio_dir.glob("*.wav"))
    if is_positive and require_segs_for_positives:
        wavs = [w for w in wavs if (audio_dir / f"{w.stem}_segs.txt").exists()]
    if max_files:
        wavs = wavs[:max_files]
    for wp in wavs:
        if "_speech" in wp.stem:
            continue
        audio = load_wav_resample_if_needed(wp)
        if audio is None or len(audio) < CHUNK_SAMPLES:
            continue
        n_chunks = len(audio) // CHUNK_SAMPLES
        if is_positive:
            segs = audio_dir / f"{wp.stem}_segs.txt"
            if segs.exists():
                labels = load_segs_to_chunk_labels(segs, n_chunks)
            else:
                labels = np.ones(n_chunks, dtype=np.int32)
        else:
            labels = np.zeros(n_chunks, dtype=np.int32)
        out.append((audio, labels, wp.stem))
    return out


def run_all_detectors(audio_items: list[tuple[np.ndarray, np.ndarray, str]],
                       run_silero: bool = True):
    """Process each detector per-file (with state reset between files), concatenate
    chunk-level scores+labels, compute metrics over combined population.
    This matches Silero's eval protocol where each clip is independent."""
    total_audio_s = sum(len(a) for a, _, _ in audio_items) / SAMPLE_RATE
    print(f"\ntotal audio across {len(audio_items)} files: {total_audio_s:.1f}s")

    configs = [
        ("energy", lambda a: energy_score_chunks(a)),
        ("webrtc_m3", lambda a: webrtc_score_chunks(a, mode=3)),
        ("lc_canon_sd4_s4", lambda a: lc_score_chunks(a, steps_per_chunk=4, sig_delta=4, kernel="canonical")),
        ("lc_simple_s8", lambda a: lc_score_chunks(a, steps_per_chunk=8, sig_delta=1, kernel="simple_avg")),
        ("lc_simple_s4", lambda a: lc_score_chunks(a, steps_per_chunk=4, sig_delta=1, kernel="simple_avg")),
        ("lc_ema_s4", lambda a: lc_score_chunks(a, steps_per_chunk=4, sig_delta=4, kernel="ema")),
    ]
    if run_silero and _SILERO_AVAILABLE:
        configs.append(("silero_v5", silero_score_chunks))
    if _TENVAD_AVAILABLE:
        configs.append(("ten_vad", ten_vad_score_chunks))

    print()
    print(f"{'detector':<15} {'ROC-AUC':>8} {'acc@0.5':>8} {'F1':>6} "
          f"{'thr':>5} {'P':>5} {'R':>5} {'us/chunk':>10} {'RTS':>8}")
    print("-" * 85)
    rows = []
    for name, fn in configs:
        all_scores, all_labels = [], []
        total_elapsed = 0.0
        total_chunks = 0
        for audio, labels, _ in audio_items:
            scores, stats = fn(audio)
            n = min(len(scores), len(labels))
            all_scores.append(scores[:n])
            all_labels.append(labels[:n])
            total_elapsed += stats.get("elapsed_s", 0.0)
            total_chunks += n
        scores = np.concatenate(all_scores) if all_scores else np.zeros(0, dtype=np.float32)
        labels = np.concatenate(all_labels) if all_labels else np.zeros(0, dtype=np.int32)
        auc = roc_auc(scores, labels)
        acc = accuracy_at_thr(scores, labels, 0.5)
        f1, thr, m = threshold_sweep(scores, labels)
        us_chunk = (total_elapsed * 1e6) / max(total_chunks, 1)
        rts = (CHUNK_MS * 1000) / max(us_chunk, 1e-9)
        print(f"{name:<15} {auc:>8.3f} {acc:>8.3f} {f1:>6.3f} "
              f"{thr:>5.2f} {m['prec']:>5.2f} {m['rec']:>5.2f} {us_chunk:>10.1f} {rts:>8.1f}x")
        rows.append({"name": name, "roc_auc": auc, "acc": acc, "f1": f1, "thr": thr,
                     "us_chunk": us_chunk, "rts": rts,
                     "positive_pct": float(labels.mean() * 100)})
    return rows


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-dir", type=Path, default=None,
                    help="Negative samples (no speech), e.g. data/esc50/audio")
    ap.add_argument("--pos-dir", type=Path, default=Path("runs/insta_audio"),
                    help="Positive samples (speech, with optional _segs.txt)")
    ap.add_argument("--max-neg", type=int, default=200)
    ap.add_argument("--max-pos", type=int, default=20)
    ap.add_argument("--no-silero", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON output path for results")
    args = ap.parse_args()

    items: list[tuple[np.ndarray, np.ndarray, str]] = []

    if args.pos_dir and args.pos_dir.exists():
        pos_items = collect_audio_chunks(args.pos_dir, is_positive=True, max_files=args.max_pos)
        print(f"positive files: {len(pos_items)} (from {args.pos_dir})")
        items.extend(pos_items)

    if args.neg_dir and args.neg_dir.exists():
        neg_items = collect_audio_chunks(args.neg_dir, is_positive=False, max_files=args.max_neg)
        print(f"negative files: {len(neg_items)} (from {args.neg_dir})")
        items.extend(neg_items)

    if not items:
        sys.exit("no audio found; pass --neg-dir and/or --pos-dir")

    rows = run_all_detectors(items, run_silero=not args.no_silero)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "chunk_ms": CHUNK_MS,
            "chunk_samples": CHUNK_SAMPLES,
            "sample_rate": SAMPLE_RATE,
            "n_files": len(items),
            "neg_dir": str(args.neg_dir) if args.neg_dir else None,
            "pos_dir": str(args.pos_dir) if args.pos_dir else None,
            "platform": sys.platform,
            "results": rows,
        }, indent=2))
        print(f"\nresults saved → {args.out}")


if __name__ == "__main__":
    main()
