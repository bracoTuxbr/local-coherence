// runtime.hpp — affinity, priority, calibrated TSC, topology
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <thread>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

#if defined(_MSC_VER)
  #include <intrin.h>
#else
  #include <x86intrin.h>
#endif

#ifndef _WIN32
  #include <pthread.h>
  #include <sched.h>
#endif

namespace cl {

// === Time (TSC calibrated against QPC) ============================
struct TscClock {
    double ns_per_tick = 0.0;     // calibrated once at startup
    uint64_t qpc_freq  = 0;
    bool     invariant = false;   // CPUID 0x80000007:edx bit 8 (AMD)

    void calibrate() {
#ifdef _WIN32
        LARGE_INTEGER f, t0, t1;
        QueryPerformanceFrequency(&f);
        qpc_freq = (uint64_t)f.QuadPart;
        unsigned aux;
        QueryPerformanceCounter(&t0);
        uint64_t r0 = __rdtscp(&aux);
        // ~50 ms calibration window
        Sleep(50);
        uint64_t r1 = __rdtscp(&aux);
        QueryPerformanceCounter(&t1);
        double secs = double(t1.QuadPart - t0.QuadPart) / double(qpc_freq);
        double ticks = double(r1 - r0);
        ns_per_tick = (secs * 1e9) / ticks;

        int regs[4] = {0,0,0,0};
#if defined(_MSC_VER)
        __cpuid(regs, 0x80000007);
#else
        __asm__ __volatile__("cpuid"
            : "=a"(regs[0]),"=b"(regs[1]),"=c"(regs[2]),"=d"(regs[3])
            : "a"(0x80000007),"c"(0));
#endif
        invariant = (regs[3] & (1u << 8)) != 0;
#else
        // simplified POSIX calibration
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        unsigned aux;
        uint64_t r0 = __rdtscp(&aux);
        struct timespec d{0, 50'000'000};
        nanosleep(&d, nullptr);
        uint64_t r1 = __rdtscp(&aux);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double secs = double(t1.tv_sec - t0.tv_sec) + double(t1.tv_nsec - t0.tv_nsec)*1e-9;
        ns_per_tick = (secs * 1e9) / double(r1 - r0);
        invariant = true;
#endif
    }

    static inline uint64_t now() {
        unsigned aux;
        return __rdtscp(&aux);
    }
    inline double to_ns(uint64_t ticks) const { return double(ticks) * ns_per_tick; }
};

// barrier against TSC read reordering by the compiler/CPU
static inline void cpu_serialize() {
#if defined(_MSC_VER)
    int regs[4]; __cpuid(regs, 0);
#else
    unsigned a, b, c, d;
    __asm__ __volatile__("cpuid" : "=a"(a),"=b"(b),"=c"(c),"=d"(d) : "a"(0));
#endif
}

// === Affinity and priority ==========================================
inline bool pin_thread_to_core(int logical_index) {
#ifdef _WIN32
    DWORD_PTR mask = (DWORD_PTR)1ull << logical_index;
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
#else
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(logical_index, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#endif
}

inline void boost_thread_priority() {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
#endif
}

inline void boost_process_priority() {
#ifdef _WIN32
    // WARNING: REALTIME can hang the OS if the loop never yields. Use with caution.
    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
        // no privilege, ignore
    }
#endif
}

// === Topology (CPUID) ==============================================
struct CpuInfo {
    char     vendor[13]    = {0};
    char     brand[49]     = {0};
    uint32_t family        = 0;
    uint32_t model         = 0;
    uint32_t stepping      = 0;
    int      logical_count = 0;
    int      physical_count = 0;
    bool     smt_on        = false;
    bool     has_avx2      = false;
    bool     has_avx512f   = false;
    bool     has_invariant_tsc = false;
};

inline void cpuid_raw(int leaf, int subleaf, int regs[4]) {
#if defined(_MSC_VER)
    __cpuidex(regs, leaf, subleaf);
#else
    __asm__ __volatile__("cpuid"
        : "=a"(regs[0]),"=b"(regs[1]),"=c"(regs[2]),"=d"(regs[3])
        : "a"(leaf),"c"(subleaf));
#endif
}

inline CpuInfo detect_cpu() {
    CpuInfo i{};
    int r[4];

    // memcpy avoids strict aliasing UB (reading r[k] as int and writing
    // i.vendor/i.brand as char*). Modern compilers optimize to a direct mov.
    cpuid_raw(0, 0, r);
    std::memcpy(&i.vendor[0], &r[1], sizeof(int));
    std::memcpy(&i.vendor[4], &r[3], sizeof(int));
    std::memcpy(&i.vendor[8], &r[2], sizeof(int));
    i.vendor[12] = 0;

    cpuid_raw(0x80000002, 0, r); for (int k=0;k<4;k++) std::memcpy(&i.brand[k*4],     &r[k], sizeof(int));
    cpuid_raw(0x80000003, 0, r); for (int k=0;k<4;k++) std::memcpy(&i.brand[16+k*4],  &r[k], sizeof(int));
    cpuid_raw(0x80000004, 0, r); for (int k=0;k<4;k++) std::memcpy(&i.brand[32+k*4],  &r[k], sizeof(int));
    i.brand[48] = 0;

    cpuid_raw(1, 0, r);
    uint32_t a = (uint32_t)r[0];
    uint32_t base_family = (a >> 8) & 0xF;
    uint32_t base_model  = (a >> 4) & 0xF;
    uint32_t ext_family  = (a >> 20) & 0xFF;
    uint32_t ext_model   = (a >> 16) & 0xF;
    i.family   = (base_family == 0xF) ? (base_family + ext_family) : base_family;
    i.model    = (base_family == 0x6 || base_family == 0xF) ? ((ext_model << 4) | base_model) : base_model;
    i.stepping = a & 0xF;

    cpuid_raw(7, 0, r);
    i.has_avx2     = (r[1] & (1u << 5))  != 0;
    i.has_avx512f  = (r[1] & (1u << 16)) != 0;

    cpuid_raw(0x80000007, 0, r);
    i.has_invariant_tsc = (r[3] & (1u << 8)) != 0;

    i.logical_count  = (int)std::thread::hardware_concurrency();

#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len > 0) {
        std::vector<uint8_t> buf(len);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)buf.data(), &len)) {
            int physical = 0;
            uint8_t* p = buf.data(); uint8_t* end = p + len;
            while (p < end) {
                auto* e = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)p;
                if (e->Relationship == RelationProcessorCore) physical++;
                p += e->Size;
            }
            i.physical_count = physical;
        }
    }
    if (i.physical_count == 0) i.physical_count = i.logical_count;
    i.smt_on = (i.logical_count > i.physical_count);
#else
    i.physical_count = i.logical_count;
#endif
    return i;
}

// returns a list of logical indices corresponding to 1 thread per physical core.
// On Windows we use GetLogicalProcessorInformationEx to map cores.
inline std::vector<int> physical_core_indices() {
    std::vector<int> out;
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return out;
    std::vector<uint8_t> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)buf.data(), &len)) return out;
    uint8_t* p = buf.data(); uint8_t* end = p + len;
    while (p < end) {
        auto* e = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)p;
        if (e->Relationship == RelationProcessorCore) {
            // first logical of that core
            const auto& g = e->Processor.GroupMask[0];
            for (int b = 0; b < 64; ++b) {
                if (g.Mask & (1ull << b)) { out.push_back(b); break; }
            }
        }
        p += e->Size;
    }
#else
    int n = (int)std::thread::hardware_concurrency();
    for (int i = 0; i < n; ++i) out.push_back(i);
#endif
    return out;
}

inline void print_cpu(const CpuInfo& i) {
    std::printf("vendor=%s\n", i.vendor);
    std::printf("brand=%s\n", i.brand);
    std::printf("family=0x%X model=0x%X stepping=%u\n", i.family, i.model, i.stepping);
    std::printf("logical=%d physical=%d smt=%s\n",
        i.logical_count, i.physical_count, i.smt_on ? "on" : "off");
    std::printf("avx2=%s avx512f=%s invariant_tsc=%s\n",
        i.has_avx2 ? "yes" : "no",
        i.has_avx512f ? "yes" : "no",
        i.has_invariant_tsc ? "yes" : "no");
}

} // namespace cl
