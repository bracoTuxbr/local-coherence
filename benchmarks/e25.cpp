// e25.cpp — Sprint v0 #59 (H4 dossier): robust detector on real mel
//
// Question: does the adaptive runtime M5.5 (probe every 16 gens, hysteresis 25%/40%)
// remain stable on real signals (user's Instagram audio) without oscillating
// pathologically between dirty and naive?
//
// Criterion: <= 8 transitions / 30s audio (from H4 dossier).
// If transitions > 30, current hysteresis is insufficient for real signal.
//
// Procedure:
//   1. Load N WAVs from runs/insta_audio/
//   2. For each: pipeline_run -> mel_buf[n_mel x n_frames]
//   3. Initialize HotField2D with mel
//   4. Run adaptive_pass for K gens
//   5. Count transitions (mode swaps), gens in each mode
//   6. Report: mean and max transitions, robust if <= 8 / sample

#include "runtime.hpp"
#include "tissue2d.hpp"
#include "diffuse2d.hpp"
#include "dirty2d.hpp"
#include "propagate_dirty2d.hpp"
#include "adaptive.hpp"
#include "wav.hpp"
#include "audio_pipeline.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace cl;

struct WavRunResult {
    std::string name;
    int n_frames;
    double duration_s;
    int gens_dirty;
    int gens_naive;
    int gens_probe;
    int transitions;
};

static WavRunResult run_one_wav(const std::string& wav_path, int gens) {
    WavRunResult r{wav_path, 0, 0, 0, 0, 0, 0};

    WavData wav;
    if (!wav_read_pcm16(wav_path, wav)) {
        std::fprintf(stderr, "FAIL read %s\n", wav_path.c_str());
        return r;
    }

    AudioPipelineConfig cfg;
    AudioPipelineState ps;
    pipeline_init(ps, cfg);

    std::vector<uint16_t> mel_buf;
    int n_frames = 0;
    pipeline_run(ps, wav, mel_buf, n_frames);
    if (n_frames < 50) {
        std::fprintf(stderr, "SKIP %s (n_frames=%d too short)\n",
                     wav_path.c_str(), n_frames);
        return r;
    }
    r.n_frames = n_frames;
    r.duration_s = (double)wav.samples.size() / (double)cfg.sample_rate;

    int n_mel = cfg.n_mel;

    // 2D tissue: rows = n_mel, cols = n_frames
    DirtyTissue2D d;
    d.H = (size_t)n_mel;
    d.W = (size_t)n_frames;
    size_t TR = TILE_R, TC = TILE_C;
    d.NTR = (d.H + TR - 1) / TR;
    d.NTC = (d.W + TC - 1) / TC;
    d.n_tiles = d.NTR * d.NTC;
    d.n_words = (d.n_tiles + 63) / 64;
    d.a = alloc_hf2d(d.H, d.W, 1);
    d.b = alloc_hf2d(d.H, d.W, 1);
    d.stability = (uint8_t*)std::calloc(d.H * d.W, 1);
    d.stab_stride = d.W;
    d.dirty = (uint64_t*)std::calloc(d.n_words, sizeof(uint64_t));
    auto next_dirty = (uint64_t*)std::calloc(d.n_words, sizeof(uint64_t));
    if (!d.a.base || !d.b.base || !d.stability || !d.dirty || !next_dirty) {
        std::fprintf(stderr, "FAIL tissue alloc for %s\n", wav_path.c_str());
        return r;
    }

    // initialize tissue with mel
    clear_hf2d(d.a); clear_hf2d(d.b);
    for (int m = 0; m < n_mel; ++m) {
        for (int f = 0; f < n_frames; ++f) {
            uint16_t v = mel_buf[(size_t)m * (size_t)n_frames + (size_t)f];
            d.a.data[(size_t)m * d.a.stride + (size_t)f] = v;
            // mark all tiles dirty initially
        }
    }
    std::memset(d.dirty, 0xFF, d.n_words * sizeof(uint64_t));

    // adaptive runtime
    AdaptiveState st;
    bool prev_is_a = true;
    for (int g = 0; g < gens; ++g) {
        size_t active = 0;
        adaptive_pass(d, st, prev_is_a, g, next_dirty, &active);
        apply_next_dirty2d(d, next_dirty);
        prev_is_a = !prev_is_a;
    }

    r.gens_dirty = st.gens_dirty;
    r.gens_naive = st.gens_naive;
    r.gens_probe = st.gens_probe;
    r.transitions = st.transitions;

    free_hf2d(d.a); free_hf2d(d.b);
    std::free(d.stability); std::free(d.dirty); std::free(next_dirty);

    return r;
}

int main(int argc, char** argv) {
    int gens = (argc > 1) ? std::atoi(argv[1]) : 200;

    auto info = detect_cpu();
    std::fprintf(stderr, "=== local-coherence / E25 detector M5.5 on real mel (#59) ===\n");
    print_cpu(info);
    std::fprintf(stderr, "gens=%d  hysteresis=25/40%%  probe_interval=16\n\n", gens);

    boost_process_priority();
    pin_thread_to_core(0);
    boost_thread_priority();

    // list WAVs from runs/insta_audio
    std::vector<std::string> wavs = {
        "C--y4BQPJhR.wav", "C-frp-QxQLm.wav", "C2sPWsUO08s.wav",
        "C3d0FV2uMBS.wav", "C5O6AW2pX7t.wav", "CeG33K1gC8U.wav",
        "Cl40MstAqMK.wav", "Clg9KJoAa7R.wav", "CptCta8PyUj.wav",
        "CsFUZ2eL9hL.wav"
    };
    std::string base = "C:/Users/ThiagoAlencar/Documents/IA/coerencia-local/runs/insta_audio/";

    std::printf("wav,n_frames,duration_s,gens_dirty,gens_naive,gens_probe,transitions,robust\n");
    std::fprintf(stderr,
        "  wav                       n_frames  dur(s)  d/n/p          transitions  robust(<=8)?\n");

    int total_runs = 0;
    int total_transitions = 0;
    int max_transitions = 0;
    int n_robust = 0;
    int n_skipped = 0;

    for (const auto& wname : wavs) {
        auto r = run_one_wav(base + wname, gens);
        if (r.n_frames < 50) { ++n_skipped; continue; }
        ++total_runs;
        total_transitions += r.transitions;
        if (r.transitions > max_transitions) max_transitions = r.transitions;
        // H4 criterion: <= 8 transitions / 30s audio. adjust for wav duration:
        // budget = 8 * (duration_s / 30)
        double budget = 8.0 * (r.duration_s / 30.0);
        if (budget < 4) budget = 4; // minimum 4 transitions allowed
        bool robust = r.transitions <= (int)(budget + 0.5);
        if (robust) ++n_robust;
        const char* rob_str = robust ? "YES" : "NO";

        std::fprintf(stderr,
            "  %-25s %-9d %-7.2f %4d/%4d/%4d  %-12d %s\n",
            wname.c_str(), r.n_frames, r.duration_s,
            r.gens_dirty, r.gens_naive, r.gens_probe, r.transitions, rob_str);
        std::printf("%s,%d,%.2f,%d,%d,%d,%d,%s\n",
            wname.c_str(), r.n_frames, r.duration_s,
            r.gens_dirty, r.gens_naive, r.gens_probe, r.transitions, rob_str);
    }

    std::fprintf(stderr, "\n=== Summary ===\n");
    std::fprintf(stderr, "  total runs       = %d (skipped %d short)\n", total_runs, n_skipped);
    if (total_runs > 0) {
        double avg_t = (double)total_transitions / (double)total_runs;
        double robust_pct = 100.0 * (double)n_robust / (double)total_runs;
        std::fprintf(stderr, "  avg transitions  = %.1f / sample\n", avg_t);
        std::fprintf(stderr, "  max transitions  = %d / sample\n", max_transitions);
        std::fprintf(stderr, "  robust samples   = %d/%d (%.0f%%)\n", n_robust, total_runs, robust_pct);
        std::fprintf(stderr, "\n");
        if (n_robust == total_runs) {
            std::fprintf(stderr, "PASS: detector M5.5 robust on all real samples.\n");
            std::fprintf(stderr, "Hysteresis 25/40%% sufficient for Instagram audio (real, noisy signal).\n");
        } else if (robust_pct >= 80.0) {
            std::fprintf(stderr, "PARTIAL: %d/%d robust samples. Hysteresis works on majority.\n",
                         n_robust, total_runs);
        } else {
            std::fprintf(stderr, "FAIL: detector breaks on %d/%d samples.\n",
                         total_runs - n_robust, total_runs);
            std::fprintf(stderr, "Current hysteresis insufficient; consider EMA + dynamic threshold.\n");
            return 1;
        }
    }
    return 0;
}
