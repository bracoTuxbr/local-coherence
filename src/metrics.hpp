// metrics.hpp — helpers reusaveis para medir comportamento do paradigma
//
// Nao-CORE. Uso livre em experimentos. Funcoes deterministicas, sem state
// global. Metricas:
//   - active_count_1d / active_count_2d: |A(t)|
//   - r_eff_1d: raio efetivo a partir de um centro
//   - run_until_stable_1d: propaga ate bitmap dirty == 0 (M-3)
//   - run_until_no_change_1d: propaga ate prev == next bit-exact
//   - propagation_trace_1d: trace de |A(t)| e r_eff(t) por gen (M-4)
//
// Para o paradigma, "estavel" tem 3 definicoes possiveis:
//   1. bitmap dirty completamente zero (definicao do paradigma)
//   2. campo all-zero (estado quente esfriou)
//   3. prev == next (convergencia bit-exact)
// Usamos (1) como primaria; (3) como auxiliar.

#pragma once

#include "propagate.hpp"
#include "tissue2d.hpp"
#include "diffuse2d.hpp"
#include "dirty.hpp"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

namespace cl {

// === ATIVIDADE: |A(t)| ===

inline size_t active_count_1d(const HotField16& h) {
    size_t c = 0;
    for (size_t i = 0; i < h.n; ++i) if (h.data[i] > 0) ++c;
    return c;
}

inline size_t active_count_2d(const HotField2D& f) {
    size_t c = 0;
    for (size_t r = 0; r < f.H; ++r) {
        const uint16_t* row = f.data + r * f.stride;
        for (size_t k = 0; k < f.W; ++k) if (row[k] > 0) ++c;
    }
    return c;
}

// === RAIO EFETIVO ===
// Maior k tal que data[center + k] > 0 OU data[center - k] > 0.
// Para campos com perturbacao centrada (pulse, source).
inline size_t r_eff_1d(const HotField16& h, size_t center) {
    if (center >= h.n) return 0;
    size_t lr = (center < h.n - center - 1) ? center : (h.n - center - 1);
    size_t r = 0;
    for (size_t k = 0; k <= lr; ++k) {
        if ((center + k < h.n && h.data[center + k] > 0) ||
            (center >= k && h.data[center - k] > 0)) {
            r = k;
        }
    }
    return r;
}

// === ESTABILIZACAO ===

struct StableResult {
    int    gen_stable;       // gen em que ficou estavel; -1 se nao convergiu
    size_t gens_with_activity; // total gens em que |A|>0
    size_t peak_active;      // max |A(t)|
    int    peak_gen;
    size_t peak_r_eff;
};

// Roda ate prev == next bit-exact (definicao 3). Caller fornece prev/next ja
// inicializados; assume halo zerado externamente em cada gen.
// Retorna gen em que prev == next (zero gens significa ja estavel inicialmente).
// Se nao convergir em max_gens, gen_stable = -1.
inline StableResult run_until_no_change_1d(HotField16& a, HotField16& b,
                                           int max_gens, size_t center)
{
    StableResult res = {-1, 0, 0, 0, 0};
    HotField16* prev = &a;
    HotField16* next = &b;
    {
        size_t ac = active_count_1d(*prev);
        if (ac > 0) ++res.gens_with_activity;
        if (ac > res.peak_active) { res.peak_active = ac; res.peak_gen = 0; }
        size_t rf = r_eff_1d(*prev, center);
        if (rf > res.peak_r_eff) res.peak_r_eff = rf;
    }
    for (int g = 1; g <= max_gens; ++g) {
        prev->data[-1] = 0; prev->data[prev->n] = 0;
        propagate_1d(*prev, *next);
        size_t ac = active_count_1d(*next);
        if (ac > 0) ++res.gens_with_activity;
        if (ac > res.peak_active) { res.peak_active = ac; res.peak_gen = g; }
        size_t rf = r_eff_1d(*next, center);
        if (rf > res.peak_r_eff) res.peak_r_eff = rf;
        // bit-exact comparison
        if (std::memcmp(prev->data, next->data, prev->n * sizeof(uint16_t)) == 0) {
            res.gen_stable = g;
            return res;
        }
        std::swap(prev, next);
    }
    return res;
}

// === TRACE M-4 ===

struct PropagationTrace {
    std::vector<size_t> active;  // |A(t)| em cada gen, [0..max_gens]
    std::vector<size_t> r_eff;   // r_eff(t) em cada gen
};

inline PropagationTrace propagation_trace_1d(HotField16& a, HotField16& b,
                                             int max_gens, size_t center)
{
    PropagationTrace tr;
    tr.active.reserve(max_gens + 1);
    tr.r_eff.reserve(max_gens + 1);
    HotField16* prev = &a;
    HotField16* next = &b;
    tr.active.push_back(active_count_1d(*prev));
    tr.r_eff.push_back(r_eff_1d(*prev, center));
    for (int g = 1; g <= max_gens; ++g) {
        prev->data[-1] = 0; prev->data[prev->n] = 0;
        propagate_1d(*prev, *next);
        tr.active.push_back(active_count_1d(*next));
        tr.r_eff.push_back(r_eff_1d(*next, center));
        std::swap(prev, next);
    }
    return tr;
}

inline PropagationTrace propagation_trace_2d(HotField2D& a, HotField2D& b,
                                             int max_gens)
{
    PropagationTrace tr;
    tr.active.reserve(max_gens + 1);
    tr.r_eff.reserve(max_gens + 1);
    HotField2D* prev = &a;
    HotField2D* next = &b;
    tr.active.push_back(active_count_2d(*prev));
    tr.r_eff.push_back(0);
    for (int g = 1; g <= max_gens; ++g) {
        diffuse2d_pass(*prev, *next);
        tr.active.push_back(active_count_2d(*next));
        tr.r_eff.push_back(0);
        std::swap(prev, next);
    }
    return tr;
}

} // namespace cl
