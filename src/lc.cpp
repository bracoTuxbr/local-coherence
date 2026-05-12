// lc.cpp — implementation of public C99 API declared in include/lc/lc.h
//
// Strategy: opaque struct lc_tissue holds union of 1D and 2D state. Each
// extern "C" function dispatches based on dimensionality flag set at create.
// Internal C++ types (HotField16, DirtyTissue, etc) are not exposed.

#include "lc/lc.h"

#include "tissue.hpp"
#include "tissue2d.hpp"
#include "propagate.hpp"
#include "propagate_dirty.hpp"
#include "dirty.hpp"
#include "dirty2d.hpp"
#include "propagate_dirty2d.hpp"
#include "diffuse2d.hpp"
#include "adaptive.hpp"
#include "metrics.hpp"

#include <cstdlib>
#include <cstring>
#include <new>

using namespace cl;

namespace {

enum lc_dim_t { LC_1D = 1, LC_2D = 2 };

// Per-instance config + state union.
struct State1D {
    DirtyTissue d;
    uint64_t* next_bits;
    bool prev_is_a;
    AdaptiveState ad;
    bool from_external_buffer;  // if true, base must not be freed
};

struct State2D {
    DirtyTissue2D d;
    uint64_t* next_bits;
    bool prev_is_a;
    AdaptiveState ad;
};

} // anonymous

struct lc_tissue {
    lc_dim_t dim;
    int n_threads;          // future: used by MT path
    uint16_t pulse_default;
    lc_kernel_id_t kernel_id;   // ABI 1.2: kernel selection (1D only)
    State1D s1;
    State2D s2;
};

extern "C" {

// === Lifecycle ===

lc_tissue_t* lc_create_1d(size_t n_cells) {
    if (n_cells == 0) return nullptr;
    lc_tissue_t* t = (lc_tissue_t*)std::calloc(1, sizeof(lc_tissue_t));
    if (!t) return nullptr;
    t->dim = LC_1D;
    t->n_threads = 1;
    t->pulse_default = 60000;
    t->s1.d.n = n_cells;
    t->s1.d.n_chunks = (n_cells + CHUNK_CELLS - 1) / CHUNK_CELLS;
    t->s1.d.n_words = (t->s1.d.n_chunks + 63) / 64;
    t->s1.d.a = alloc_hf16(n_cells, 1);
    t->s1.d.b = alloc_hf16(n_cells, 1);
    t->s1.d.stability = (uint8_t*)std::calloc(n_cells, 1);
    t->s1.d.dirty = (uint64_t*)std::calloc(t->s1.d.n_words, sizeof(uint64_t));
    t->s1.next_bits = (uint64_t*)std::calloc(t->s1.d.n_words, sizeof(uint64_t));
    t->s1.prev_is_a = true;
    t->s1.from_external_buffer = false;
    if (!t->s1.d.a.base || !t->s1.d.b.base || !t->s1.d.stability ||
        !t->s1.d.dirty || !t->s1.next_bits) {
        lc_destroy(t);
        return nullptr;
    }
    for (size_t i = 0; i < n_cells; ++i) {
        t->s1.d.a.data[i] = 0;
        t->s1.d.b.data[i] = 0;
    }
    t->s1.ad = AdaptiveState();
    return t;
}

lc_tissue_t* lc_create_2d(size_t H, size_t W) {
    if (H == 0 || W == 0) return nullptr;
    lc_tissue_t* t = (lc_tissue_t*)std::calloc(1, sizeof(lc_tissue_t));
    if (!t) return nullptr;
    t->dim = LC_2D;
    t->n_threads = 1;
    t->pulse_default = 60000;
    t->s2.d.H = H; t->s2.d.W = W;
    t->s2.d.NTR = (H + TILE_R - 1) / TILE_R;
    t->s2.d.NTC = (W + TILE_C - 1) / TILE_C;
    t->s2.d.n_tiles = t->s2.d.NTR * t->s2.d.NTC;
    t->s2.d.n_words = (t->s2.d.n_tiles + 63) / 64;
    t->s2.d.a = alloc_hf2d(H, W, 1);
    t->s2.d.b = alloc_hf2d(H, W, 1);
    t->s2.d.stability = (uint8_t*)std::calloc(H * W, 1);
    t->s2.d.stab_stride = W;
    t->s2.d.dirty = (uint64_t*)std::calloc(t->s2.d.n_words, sizeof(uint64_t));
    t->s2.next_bits = (uint64_t*)std::calloc(t->s2.d.n_words, sizeof(uint64_t));
    t->s2.prev_is_a = true;
    if (!t->s2.d.a.base || !t->s2.d.b.base || !t->s2.d.stability ||
        !t->s2.d.dirty || !t->s2.next_bits) {
        lc_destroy(t);
        return nullptr;
    }
    clear_hf2d(t->s2.d.a);
    clear_hf2d(t->s2.d.b);
    /* Init AdaptiveState explicitly: calloc() zeroed bytes ignore C++
     * default member initializers, leaving probe_interval=0 (div by zero). */
    t->s2.ad = AdaptiveState();
    return t;
}

lc_tissue_t* lc_create_from_buffer_1d(const uint16_t* buf, size_t n) {
    if (!buf || n == 0) return nullptr;
    lc_tissue_t* t = lc_create_1d(n);
    if (!t) return nullptr;
    // copia conteudo do buf para o tecido (NAO compartilha memoria por
    // halo + double-buffer reasons; behaviour documented in ABI.md)
    for (size_t i = 0; i < n; ++i) {
        t->s1.d.a.data[i] = buf[i];
    }
    // mark everything dirty so the first step processes the whole tissue
    std::memset(t->s1.d.dirty, 0xFF, t->s1.d.n_words * sizeof(uint64_t));
    return t;
}

void lc_destroy(lc_tissue_t* t) {
    if (!t) return;
    if (t->dim == LC_1D) {
        free_hf16(t->s1.d.a); free_hf16(t->s1.d.b);
        if (t->s1.d.stability) std::free(t->s1.d.stability);
        if (t->s1.d.dirty) std::free(t->s1.d.dirty);
        if (t->s1.next_bits) std::free(t->s1.next_bits);
    } else if (t->dim == LC_2D) {
        free_hf2d(t->s2.d.a); free_hf2d(t->s2.d.b);
        if (t->s2.d.stability) std::free(t->s2.d.stability);
        if (t->s2.d.dirty) std::free(t->s2.d.dirty);
        if (t->s2.next_bits) std::free(t->s2.next_bits);
    }
    std::free(t);
}

// === Configure ===

void lc_set_threads(lc_tissue_t* t, int n) {
    if (!t || n < 1) return;
    t->n_threads = n;  // futuro: integrar com multi-thread workers (M19/M20)
}

void lc_set_pulse_value(lc_tissue_t* t, uint16_t v) {
    if (!t) return;
    t->pulse_default = v;
}

void lc_set_sig_delta(lc_tissue_t* t, uint16_t v) {
    if (!t || t->dim != LC_1D) return;
    t->s1.d.sig_delta = v;
}

void lc_set_kernel(lc_tissue_t* t, lc_kernel_id_t id) {
    if (!t || t->dim != LC_1D) return;
    t->kernel_id = id;
}

// === Inject ===

void lc_inject_1d(lc_tissue_t* t, size_t pos, size_t len, uint16_t v) {
    if (!t || t->dim != LC_1D) return;
    HotField16& cur = t->s1.prev_is_a ? t->s1.d.a : t->s1.d.b;
    size_t end = pos + len;
    if (end > t->s1.d.n) end = t->s1.d.n;
    if (pos > end) return;
    for (size_t i = pos; i < end; ++i) {
        cur.data[i] = v;
        t->s1.d.stability[i] = 0;
    }
    size_t cs = pos / CHUNK_CELLS;
    size_t ce = (end + CHUNK_CELLS - 1) / CHUNK_CELLS;
    for (size_t c = cs; c < ce && c < t->s1.d.n_chunks; ++c)
        chunk_set_dirty(t->s1.d.dirty, c);
    if (cs > 0) chunk_set_dirty(t->s1.d.dirty, cs - 1);
    if (ce < t->s1.d.n_chunks) chunk_set_dirty(t->s1.d.dirty, ce);
}

void lc_inject_2d(lc_tissue_t* t, size_t row, size_t col, uint16_t v) {
    if (!t || t->dim != LC_2D) return;
    if (row >= t->s2.d.H || col >= t->s2.d.W) return;
    HotField2D& cur = t->s2.prev_is_a ? t->s2.d.a : t->s2.d.b;
    cur.data[row * cur.stride + col] = v;
    t->s2.d.stability[row * t->s2.d.stab_stride + col] = 0;
    size_t tr = row / TILE_R;
    size_t tc = col / TILE_C;
    size_t ti = tr * t->s2.d.NTC + tc;
    if (ti < t->s2.d.n_tiles) chunk_set_dirty(t->s2.d.dirty, ti);
}

// === Step ===

void lc_step(lc_tissue_t* t, int n_gens) {
    if (!t || n_gens <= 0) return;
    if (t->dim == LC_1D) {
        // ABI 1.2: dispatch por kernel_id. Default LC_KERNEL_CANONICAL
        // keeps bit-exact parity with pre-1.2 baselines.
        for (int g = 0; g < n_gens; ++g) {
            switch (t->kernel_id) {
                case LC_KERNEL_SIMPLE_AVG:
                    propagate_dirty_pass_t<uint16_t, SimpleAvgKernelT<uint16_t>>(
                        t->s1.d, t->s1.prev_is_a, t->s1.next_bits);
                    break;
                case LC_KERNEL_EMA:
                    propagate_dirty_pass_t<uint16_t, EmaKernelT<uint16_t>>(
                        t->s1.d, t->s1.prev_is_a, t->s1.next_bits);
                    break;
                case LC_KERNEL_CANONICAL:
                default:
                    propagate_dirty_pass(t->s1.d, t->s1.prev_is_a, t->s1.next_bits);
                    break;
            }
            apply_next_dirty(t->s1.d, t->s1.next_bits);
            t->s1.prev_is_a = !t->s1.prev_is_a;
        }
    } else if (t->dim == LC_2D) {
        for (int g = 0; g < n_gens; ++g) {
            propagate_dirty2d_pass(t->s2.d, t->s2.prev_is_a, t->s2.next_bits);
            apply_next_dirty2d(t->s2.d, t->s2.next_bits);
            t->s2.prev_is_a = !t->s2.prev_is_a;
        }
    }
}

void lc_step_adaptive(lc_tissue_t* t, int n_gens) {
    if (!t || n_gens <= 0) return;
    if (t->dim == LC_1D) {
        // 1D adaptive is not implemented at runtime level yet.
        // Fallback para lc_step (dirty puro). Documentado em ABI.md.
        lc_step(t, n_gens);
        return;
    }
    if (t->dim == LC_2D) {
        for (int g = 0; g < n_gens; ++g) {
            size_t active = 0;
            adaptive_pass(t->s2.d, t->s2.ad, t->s2.prev_is_a, g,
                          t->s2.next_bits, &active);
            apply_next_dirty2d(t->s2.d, t->s2.next_bits);
            t->s2.prev_is_a = !t->s2.prev_is_a;
        }
    }
}

int lc_step_until_stable(lc_tissue_t* t, int max_gens) {
    if (!t || max_gens <= 0) return -1;
    if (t->dim != LC_1D) return -1;  // 2D nao suportado em v1.0
    for (int g = 1; g <= max_gens; ++g) {
        HotField16& prev = t->s1.prev_is_a ? t->s1.d.a : t->s1.d.b;
        HotField16& next = t->s1.prev_is_a ? t->s1.d.b : t->s1.d.a;
        prev.data[-1] = 0; prev.data[prev.n] = 0;
        propagate_1d(prev, next);
        if (std::memcmp(prev.data, next.data,
                        prev.n * sizeof(uint16_t)) == 0) {
            return g;
        }
        t->s1.prev_is_a = !t->s1.prev_is_a;
    }
    return -1;
}

// === Observe ===

size_t lc_active_count(const lc_tissue_t* t) {
    if (!t) return 0;
    if (t->dim == LC_1D) {
        const HotField16& cur = t->s1.prev_is_a ? t->s1.d.a : t->s1.d.b;
        return active_count_1d(cur);
    } else {
        const HotField2D& cur = t->s2.prev_is_a ? t->s2.d.a : t->s2.d.b;
        return active_count_2d(cur);
    }
}

size_t lc_r_eff(const lc_tissue_t* t) {
    if (!t || t->dim != LC_1D) return 0;
    const HotField16& cur = t->s1.prev_is_a ? t->s1.d.a : t->s1.d.b;
    return r_eff_1d(cur, cur.n / 2);
}

size_t lc_get_field(const lc_tissue_t* t, uint16_t* out, size_t max_n) {
    if (!t || !out) return 0;
    if (t->dim == LC_1D) {
        const HotField16& cur = t->s1.prev_is_a ? t->s1.d.a : t->s1.d.b;
        size_t n = (cur.n < max_n) ? cur.n : max_n;
        std::memcpy(out, cur.data, n * sizeof(uint16_t));
        return n;
    } else {
        const HotField2D& cur = t->s2.prev_is_a ? t->s2.d.a : t->s2.d.b;
        size_t total = cur.H * cur.W;
        size_t n = (total < max_n) ? total : max_n;
        // copia row-by-row (HotField2D tem stride > W)
        size_t written = 0;
        for (size_t r = 0; r < cur.H && written < n; ++r) {
            const uint16_t* row = cur.data + r * cur.stride;
            size_t to_copy = (cur.W < (n - written)) ? cur.W : (n - written);
            std::memcpy(out + written, row, to_copy * sizeof(uint16_t));
            written += to_copy;
        }
        return written;
    }
}

size_t lc_active_chunks(const lc_tissue_t* t, size_t* out, size_t max) {
    if (!t || !out) return 0;
    size_t count = 0;
    if (t->dim == LC_1D) {
        for (size_t w = 0; w < t->s1.d.n_words && count < max; ++w) {
            uint64_t word = t->s1.d.dirty[w];
            while (word && count < max) {
                int b = __builtin_ctzll(word);
                word &= word - 1;
                out[count++] = w * 64 + (size_t)b;
            }
        }
    } else {
        for (size_t w = 0; w < t->s2.d.n_words && count < max; ++w) {
            uint64_t word = t->s2.d.dirty[w];
            while (word && count < max) {
                int b = __builtin_ctzll(word);
                word &= word - 1;
                out[count++] = w * 64 + (size_t)b;
            }
        }
    }
    return count;
}

// === Diagnostics ===

void lc_abi_version(int* major, int* minor) {
    if (major) *major = LC_ABI_MAJOR;
    if (minor) *minor = LC_ABI_MINOR;
}

const char* lc_build_info(void) {
    return "lc-runtime v1.0-alpha (built " __DATE__ " " __TIME__ ")";
}

} // extern "C"
