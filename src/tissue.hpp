// tissue.hpp — 64-byte cell and tissue allocation
// goal: predictable layout, prefetch-friendly, no outbound pointers
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

namespace cl {

// Exactly 64 bytes. One x86 cache line.
// If any field grows, another must shrink. Asserted.
struct alignas(64) Cell {
    uint8_t  acoustic_type;      // 1
    uint8_t  confidence;         // 1
    uint16_t energy;             // 2
    uint16_t stability;          // 2  cycles without change >= STABLE_THRESHOLD => freeze
    uint16_t flags;              // 2  bit0=dirty, bit1=frozen, bit2=boundary
    uint32_t timestamp_delta;    // 4
    uint32_t hypothesis_a;       // 4
    uint32_t hypothesis_b;       // 4
    uint16_t score_a;            // 2
    uint16_t score_b;            // 2
    uint8_t  reserved[40];       // 40 — future use, keeps 64
};
static_assert(sizeof(Cell) == 64, "Cell must be exactly one cache line (64 bytes)");
static_assert(alignof(Cell) == 64, "Cell alignment must be 64 bytes");

enum CellFlag : uint16_t {
    FLAG_DIRTY    = 1u << 0,
    FLAG_FROZEN   = 1u << 1,
    FLAG_BOUNDARY = 1u << 2,
};

// Allocator. Tries large pages (2MB) on Windows. Fallback: VirtualAlloc + VirtualLock.
// Returns a page-aligned pointer (>= 64). nullptr on failure.
struct TissueAlloc {
    Cell*  data        = nullptr;
    size_t n_cells     = 0;
    size_t bytes       = 0;
    bool   used_lpages = false;
};

inline size_t round_up(size_t v, size_t pow2) {
    return (v + (pow2 - 1)) & ~(pow2 - 1);
}

#ifdef _WIN32
// Requires SeLockMemoryPrivilege for large pages. On failure, falls back to normal VirtualAlloc.
inline bool try_enable_lock_memory_privilege() {
    HANDLE token;
    if (!OpenProcessToken(GetCurrentProcess(),
                          TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) return false;
    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount = 1;
    if (!LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
        CloseHandle(token); return false;
    }
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    BOOL ok = AdjustTokenPrivileges(token, FALSE, &tp, 0, nullptr, nullptr);
    DWORD err = GetLastError();
    CloseHandle(token);
    return ok && err == ERROR_SUCCESS;
}
#endif

inline TissueAlloc alloc_tissue(size_t n_cells) {
    TissueAlloc a{};
    if (n_cells == 0) return a;
    // overflow check: n_cells * sizeof(Cell) must not overflow size_t
    if (n_cells > SIZE_MAX / sizeof(Cell)) return a;
    a.n_cells = n_cells;
    a.bytes   = n_cells * sizeof(Cell);

#ifdef _WIN32
    SIZE_T lp = GetLargePageMinimum();   // typically 2 MB
    if (lp > 0) {
        try_enable_lock_memory_privilege();
        SIZE_T sz = round_up(a.bytes, lp);
        void* p = VirtualAlloc(nullptr, sz,
            MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (p) {
            a.data = (Cell*)p;
            a.bytes = sz;
            a.used_lpages = true;
            return a;
        }
    }
    // fallback: normal pages + VirtualLock to avoid paging
    SIZE_T page = 4096;
    SIZE_T sz   = round_up(a.bytes, page);
    void* p = VirtualAlloc(nullptr, sz, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!p) return a;
    VirtualLock(p, sz);                  // best-effort, ignore failure
    a.data = (Cell*)p;
    a.bytes = sz;
    return a;
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, a.bytes) != 0) return a;
    a.data = (Cell*)p;
    return a;
#endif
}

inline void free_tissue(TissueAlloc& a) {
    if (!a.data) return;
#ifdef _WIN32
    VirtualUnlock(a.data, a.bytes);
    VirtualFree(a.data, 0, MEM_RELEASE);
#else
    free(a.data);
#endif
    a = {};
}

inline void touch_all(TissueAlloc& a) {
    // pre-faulting: ensures each page is resident before measurement
    volatile uint8_t sink = 0;
    uint8_t* p = (uint8_t*)a.data;
    for (size_t i = 0; i < a.bytes; i += 4096) sink ^= p[i];
    (void)sink;
}

inline void zero_init(TissueAlloc& a) {
    auto* p = (uint8_t*)a.data;
    for (size_t i = 0; i < a.bytes; ++i) p[i] = 0;
}

inline void seed_pattern(TissueAlloc& a, uint32_t seed = 0xC0FFEEu) {
    // deterministic pattern for reproducible benchmarks. xorshift32.
    uint32_t s = seed ? seed : 1u;
    for (size_t i = 0; i < a.n_cells; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        Cell& c = a.data[i];
        c.acoustic_type   = (uint8_t)(s & 0x3F);
        c.confidence      = (uint8_t)((s >> 6) & 0xFF);
        c.energy          = (uint16_t)(s & 0xFFFF);
        c.stability       = 0;
        c.flags           = FLAG_DIRTY;
        c.timestamp_delta = (uint32_t)i;
        c.hypothesis_a    = s;
        c.hypothesis_b    = s ^ 0x9E3779B1u;
        c.score_a         = (uint16_t)(s >> 16);
        c.score_b         = (uint16_t)((s ^ 0xCAFEu) & 0xFFFF);
    }
}

} // namespace cl
