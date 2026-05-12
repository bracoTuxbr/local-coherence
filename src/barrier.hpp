// barrier.hpp — sense-reversing barrier, no mutex, spin-wait with _mm_pause
//
// Why sense-reversing: avoids the classic centralized-barrier problem where
// fast threads can enter the next generation before slow threads have seen the
// release. The "sense" alternates 0/1 every generation; each thread keeps the
// sense seen in the previous generation and waits for the sense to flip.
//
// Typical cost: ~50-200 ns for 2-4 threads on the same CCD. No syscall.

#pragma once

#include <atomic>
#include <cstdint>

#if defined(_MSC_VER)
  #include <intrin.h>
#else
  #include <x86intrin.h>
#endif

namespace cl {

struct alignas(64) SpinBarrier {
    std::atomic<int>      count{0};       // threads that have not yet arrived this generation
    std::atomic<uint32_t> sense{0};       // alternates 0/1 on each release
    int                   n_threads{0};
    char                  pad[64 - 2*sizeof(std::atomic<int>) - sizeof(int)];

    void init(int n) {
        n_threads = n;
        count.store(n, std::memory_order_relaxed);
        sense.store(0, std::memory_order_relaxed);
    }

    // each thread keeps a local_sense; passed by ref and flipped by the function
    inline void wait(uint32_t& local_sense) {
        local_sense ^= 1u;
        if (count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // last to arrive releases everyone
            count.store(n_threads, std::memory_order_relaxed);
            sense.store(local_sense, std::memory_order_release);
        } else {
            // spin until global sense flips to match local
            while (sense.load(std::memory_order_acquire) != local_sense) {
                _mm_pause();
            }
        }
    }
};

} // namespace cl
