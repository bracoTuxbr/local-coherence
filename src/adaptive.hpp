// adaptive.hpp — runtime that picks between dense (naive) and sparse (dirty) kernels
//
// ============================================================================
//  CORE PARADIGM FILE (M5.5).
//  Hysteresis 25%/40% + probe every 16 gens is canonical (beats sparse 8.71x and
//  uniform 1.48x simultaneously). Any modification must preserve every EXACT
//  golden number in benchmarks/golden_numbers.txt (run tools/regression_test.ps1).
// ============================================================================
//
// Idea: every PROBE_INTERVAL generations, force a pass through the dirty kernel
// (which already measures per-tile changes). Count % of active tiles. Apply
// hysteresis to decide the mode for the next N gens.
//
// Probe is expensive on uniform workload (3.6x worse than naive), but only 1 in N.
// N=16 -> average overhead on uniform: 1 + (3.6-1)/16 = 1.16x. Acceptable.
//
// On sparse workload, probe and dirty are equal (right kernel already), no extra
// cost. On transition, probe detects the change in < 16 gens.

#pragma once

#include "tissue2d.hpp"
#include "diffuse2d.hpp"
#include "dirty2d.hpp"
#include "propagate_dirty2d.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace cl {

enum AdaptiveMode : uint8_t {
    MODE_DIRTY = 0,
    MODE_NAIVE = 1,
};

struct AdaptiveState {
    AdaptiveMode mode           = MODE_DIRTY;
    int          probe_interval = 16;
    double       enter_naive    = 0.40;   // > 40% dirty => naive
    double       exit_naive     = 0.25;   // < 25% dirty => dirty
    int          gens_naive     = 0;       // counter (telemetry)
    int          gens_dirty     = 0;
    int          gens_probe     = 0;
    int          transitions    = 0;
};

// one adaptive pass for generation `g`. Takes external state and auxiliary bitmap.
inline void adaptive_pass(DirtyTissue2D& d, AdaptiveState& s,
                          bool prev_is_a, int g,
                          uint64_t* aux_dirty,
                          size_t* out_active_tiles)
{
    bool is_probe = (g % s.probe_interval == 0);

    if (is_probe) {
        // if we came from NAIVE, bitmap is stale since the last probe.
        // force everything dirty so the probe measures the "real regime".
        if (s.mode == MODE_NAIVE) {
            std::memset(d.dirty, 0xFF, d.n_words * sizeof(uint64_t));
        }
        size_t processed = propagate_dirty2d_pass(d, prev_is_a, aux_dirty);
        apply_next_dirty2d(d, aux_dirty);
        if (out_active_tiles) *out_active_tiles = processed;
        // measure % dirty for NEXT pass (after swap)
        size_t n_dirty_now = count_dirty_tiles(d, d.dirty);
        double pct = (double)n_dirty_now / (double)d.n_tiles;
        AdaptiveMode prev_mode = s.mode;
        if (pct > s.enter_naive)        s.mode = MODE_NAIVE;
        else if (pct < s.exit_naive)    s.mode = MODE_DIRTY;
        // between the two thresholds, keep mode (hysteresis)
        if (s.mode != prev_mode) ++s.transitions;
        ++s.gens_probe;
        return;
    }

    if (s.mode == MODE_DIRTY) {
        size_t processed = propagate_dirty2d_pass(d, prev_is_a, aux_dirty);
        apply_next_dirty2d(d, aux_dirty);
        if (out_active_tiles) *out_active_tiles = processed;
        ++s.gens_dirty;
    } else { // MODE_NAIVE
        HotField2D& prev = prev_is_a ? d.a : d.b;
        HotField2D& next = prev_is_a ? d.b : d.a;
        diffuse2d_pass(prev, next);
        // bitmap stays as-is — will be reset on the next probe
        if (out_active_tiles) *out_active_tiles = d.n_tiles; // marks "dense mode"
        ++s.gens_naive;
    }
}

} // namespace cl
