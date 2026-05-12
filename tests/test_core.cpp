// test_core.cpp — unit tests for the core components
//
// Sem framework externo. Macros TEST/CHECK simples + contador.
// Roda em ~1s, exit 0 se todos passam, exit 1 se algum falha.
//
// Testa:
//   - tissue.hpp: alloc/free, edge cases, Cell layout
//   - propagate.hpp: alloc/free hf16, edge cases, propagacao basica
//   - tissue2d.hpp / dirty2d.hpp: alloc/free 2D
//   - dirty.hpp: bitmap helpers, edge cases
//   - runtime.hpp: cpuid_raw, detect_cpu basics

#include "tissue.hpp"
#include "tissue2d.hpp"
#include "propagate.hpp"
#include "dirty.hpp"
#include "dirty2d.hpp"
#include "runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) static void name()
#define CHECK(cond, msg) do { \
    if (cond) { ++g_passed; } \
    else { ++g_failed; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); } \
} while (0)

#define RUN_TEST(name) do { \
    std::fprintf(stderr, "[run] %s\n", #name); \
    int before_failed = g_failed; \
    name(); \
    if (g_failed == before_failed) std::fprintf(stderr, "[ok ] %s\n", #name); \
    else std::fprintf(stderr, "[FAIL] %s\n", #name); \
} while (0)

using namespace cl;

// === tissue.hpp ===

TEST(test_cell_size) {
    CHECK(sizeof(Cell) == 64, "Cell must be 64 bytes");
    CHECK(alignof(Cell) == 64, "Cell alignment must be 64");
}

TEST(test_alloc_tissue_basic) {
    auto a = alloc_tissue(1000);
    CHECK(a.data != nullptr, "alloc_tissue(1000) should succeed");
    CHECK(a.n_cells == 1000, "n_cells should be 1000");
    CHECK(a.bytes >= 1000 * sizeof(Cell), "bytes should fit n_cells");
    free_tissue(a);
    CHECK(a.data == nullptr, "free_tissue should null data");
    CHECK(a.n_cells == 0, "free_tissue should reset n_cells");
}

TEST(test_alloc_tissue_zero) {
    auto a = alloc_tissue(0);
    CHECK(a.data == nullptr || a.bytes == 0, "alloc_tissue(0) should return empty");
    free_tissue(a);  // no-op
}

TEST(test_seed_pattern_deterministic) {
    auto a = alloc_tissue(100);
    auto b = alloc_tissue(100);
    seed_pattern(a, 0xDEADBEEF);
    seed_pattern(b, 0xDEADBEEF);
    bool match = true;
    for (size_t i = 0; i < 100; ++i) {
        if (a.data[i].energy != b.data[i].energy) { match = false; break; }
    }
    CHECK(match, "seed_pattern with same seed should be deterministic");
    free_tissue(a);
    free_tissue(b);
}

// === propagate.hpp ===

TEST(test_alloc_hf16_basic) {
    auto h = alloc_hf16(100, 1);
    CHECK(h.base != nullptr, "alloc_hf16(100, 1) should succeed");
    CHECK(h.data == h.base + 1, "data should be base + halo");
    CHECK(h.n == 100, "n should be 100");
    CHECK(h.halo == 1, "halo should be 1");
    CHECK(h.bytes >= (100 + 2) * sizeof(uint16_t), "bytes should fit n+2*halo");
    free_hf16(h);
    CHECK(h.base == nullptr, "free_hf16 should null base");
}

TEST(test_alloc_hf16_zero) {
    auto h = alloc_hf16(0, 1);
    CHECK(h.base == nullptr, "alloc_hf16(0, 1) should return empty");
    auto h2 = alloc_hf16(100, 0);
    CHECK(h2.base == nullptr, "alloc_hf16(100, 0) should return empty (halo=0)");
}

TEST(test_alloc_hf16_overflow) {
    // allocation size that overflows size_t — must fail gracefully
    auto h = alloc_hf16(SIZE_MAX / 2, 1);
    CHECK(h.base == nullptr, "alloc_hf16 huge size should fail without crash");
}

TEST(test_propagate_1d_basic) {
    auto a = alloc_hf16(10, 1);
    auto b = alloc_hf16(10, 1);
    CHECK(a.base != nullptr && b.base != nullptr, "alloc both fields");
    // input: pulse no centro
    for (size_t i = 0; i < 10; ++i) a.data[i] = 0;
    a.data[-1] = 0; a.data[10] = 0;
    a.data[5] = 1024;
    propagate_1d(a, b);
    // after 1 gen: rule (l + 2c + r) >> 2 * 255/256
    // cell 4: (0 + 0 + 1024) >> 2 = 256, *255/256 = 255
    // cell 5: (0 + 2048 + 0) >> 2 = 512, *255/256 = 510
    // cell 6: idem cell 4 = 255
    CHECK(b.data[4] >= 250 && b.data[4] <= 260, "cell 4 should ~255 after 1 gen");
    CHECK(b.data[5] >= 505 && b.data[5] <= 515, "cell 5 should ~510 after 1 gen");
    CHECK(b.data[6] >= 250 && b.data[6] <= 260, "cell 6 should ~255 after 1 gen");
    free_hf16(a);
    free_hf16(b);
}

TEST(test_propagate_1d_decay_to_zero) {
    auto a = alloc_hf16(50, 1);
    auto b = alloc_hf16(50, 1);
    for (size_t i = 0; i < 50; ++i) a.data[i] = 100;
    a.data[-1] = 0; a.data[50] = 0;
    HotField16* prev = &a;
    HotField16* next = &b;
    for (int g = 0; g < 5000; ++g) {
        prev->data[-1] = 0; prev->data[prev->n] = 0;
        propagate_1d(*prev, *next);
        std::swap(prev, next);
    }
    // after 5000 gens with 255/256 decay, the field must be ~0
    bool all_low = true;
    for (size_t i = 0; i < 50; ++i) if (prev->data[i] > 5) { all_low = false; break; }
    CHECK(all_low, "after 5000 gens, all values should decay near zero");
    free_hf16(a);
    free_hf16(b);
}

TEST(test_swap_hf16) {
    auto a = alloc_hf16(10, 1);
    auto b = alloc_hf16(20, 1);
    auto a_data_orig = a.data;
    auto b_data_orig = b.data;
    swap_hf16(a, b);
    CHECK(a.data == b_data_orig, "swap should exchange data ptr");
    CHECK(b.data == a_data_orig, "swap should exchange data ptr");
    CHECK(a.n == 20 && b.n == 10, "swap should exchange n");
    free_hf16(a);
    free_hf16(b);
}

// === dirty.hpp ===

TEST(test_dirty_bitmap_helpers) {
    uint64_t bits[2] = {0, 0};
    CHECK(!chunk_is_dirty(bits, 0), "fresh bitmap chunk 0 should be clean");
    CHECK(!chunk_is_dirty(bits, 63), "fresh bitmap chunk 63 should be clean");
    CHECK(!chunk_is_dirty(bits, 64), "fresh bitmap chunk 64 should be clean");
    chunk_set_dirty(bits, 5);
    CHECK(chunk_is_dirty(bits, 5), "after set, chunk 5 should be dirty");
    CHECK(!chunk_is_dirty(bits, 4), "chunk 4 should remain clean");
    CHECK(!chunk_is_dirty(bits, 6), "chunk 6 should remain clean");
    chunk_set_dirty(bits, 64);  // outro word
    CHECK(chunk_is_dirty(bits, 64), "after set, chunk 64 (word 1 bit 0) should be dirty");
    chunk_clear_dirty(bits, 5);
    CHECK(!chunk_is_dirty(bits, 5), "after clear, chunk 5 should be clean");
    CHECK(chunk_is_dirty(bits, 64), "chunk 64 should remain dirty");
}

TEST(test_alloc_dirty_basic) {
    auto d = alloc_dirty(1000);
    CHECK(d.a.base != nullptr, "alloc_dirty should succeed");
    CHECK(d.b.base != nullptr, "b should be allocated");
    CHECK(d.stability != nullptr, "stability should be allocated");
    CHECK(d.dirty != nullptr, "dirty bitmap should be allocated");
    CHECK(d.n == 1000, "n should be 1000");
    CHECK(d.n_chunks == (1000 + 63) / 64, "n_chunks should match");
    // the bitmap must be fully dirty initially
    CHECK(chunk_is_dirty(d.dirty, 0), "initial bitmap should be all dirty");
    CHECK(chunk_is_dirty(d.dirty, d.n_chunks - 1), "last chunk should be dirty");
    free_dirty(d);
    CHECK(d.a.base == nullptr, "free should null all");
}

TEST(test_alloc_dirty_zero) {
    auto d = alloc_dirty(0);
    CHECK(d.a.base == nullptr, "alloc_dirty(0) should return empty");
    free_dirty(d);  // no-op
}

// === tissue2d.hpp / dirty2d.hpp ===

TEST(test_alloc_hf2d_basic) {
    auto f = alloc_hf2d(10, 20, 1);
    CHECK(f.base != nullptr, "alloc_hf2d should succeed");
    CHECK(f.H == 10, "H should be 10");
    CHECK(f.W == 20, "W should be 20");
    CHECK(f.stride == 22, "stride should be W + 2*halo = 22");
    free_hf2d(f);
    CHECK(f.base == nullptr, "free_hf2d should null base");
}

TEST(test_alloc_hf2d_zero) {
    auto f = alloc_hf2d(0, 20, 1);
    CHECK(f.base == nullptr, "alloc_hf2d(0, ...) should return empty");
    auto f2 = alloc_hf2d(10, 0, 1);
    CHECK(f2.base == nullptr, "alloc_hf2d(..., 0) should return empty");
}

TEST(test_alloc_dirty2d_basic) {
    auto d = alloc_dirty2d(64, 64);
    CHECK(d.a.base != nullptr, "alloc_dirty2d should succeed");
    CHECK(d.NTR == 1, "64/TILE_R should be 1 tile");
    CHECK(d.NTC == 1, "64/TILE_C should be 1 tile");
    CHECK(d.n_tiles == 1, "1x1 = 1 tile");
    free_dirty2d(d);
}

// === runtime.hpp ===

TEST(test_detect_cpu) {
    auto info = detect_cpu();
    CHECK(info.logical_count > 0, "logical count should be > 0");
    CHECK(info.physical_count > 0, "physical count should be > 0");
    CHECK(std::strlen(info.brand) > 0, "brand should be set");
    CHECK(std::strlen(info.vendor) > 0, "vendor should be set");
}

TEST(test_tsc_clock_calibrate) {
    TscClock c;
    c.calibrate();
    CHECK(c.ns_per_tick > 0.0, "ns_per_tick should be positive");
    CHECK(c.ns_per_tick < 10.0, "ns_per_tick should be < 10ns (sane CPU range)");
}

TEST(test_physical_core_indices) {
    auto cores = physical_core_indices();
    CHECK(cores.size() > 0, "should detect at least 1 physical core");
}

// === main runner ===

int main() {
    std::fprintf(stderr, "=== test_core: unit tests for core components ===\n\n");

    RUN_TEST(test_cell_size);
    RUN_TEST(test_alloc_tissue_basic);
    RUN_TEST(test_alloc_tissue_zero);
    RUN_TEST(test_seed_pattern_deterministic);

    RUN_TEST(test_alloc_hf16_basic);
    RUN_TEST(test_alloc_hf16_zero);
    RUN_TEST(test_alloc_hf16_overflow);
    RUN_TEST(test_propagate_1d_basic);
    RUN_TEST(test_propagate_1d_decay_to_zero);
    RUN_TEST(test_swap_hf16);

    RUN_TEST(test_dirty_bitmap_helpers);
    RUN_TEST(test_alloc_dirty_basic);
    RUN_TEST(test_alloc_dirty_zero);

    RUN_TEST(test_alloc_hf2d_basic);
    RUN_TEST(test_alloc_hf2d_zero);
    RUN_TEST(test_alloc_dirty2d_basic);

    RUN_TEST(test_detect_cpu);
    RUN_TEST(test_tsc_clock_calibrate);
    RUN_TEST(test_physical_core_indices);

    std::fprintf(stderr, "\n=== resumo ===\n");
    std::fprintf(stderr, "passed: %d\n", g_passed);
    std::fprintf(stderr, "failed: %d\n", g_failed);
    if (g_failed > 0) {
        std::fprintf(stderr, "FAIL\n");
        return 1;
    }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
