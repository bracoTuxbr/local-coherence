# Security policy

> **Status note**: a focused security review of the CORE files was run on
> **2026-05-07** (the original "Sprint v0" baseline). Several issues were
> identified and fixed at that point (see "Hardening done" below). The
> codebase has evolved since: pluggable kernels (ABI 1.2), the
> `HotField<T>` template generalization, multiple refactors. **A
> repeat audit of the post-Sprint-v0 surface has not yet been performed.**
> This document describes the threat model and the original hardening; it
> is *not* a current-state attestation.

## Threat model

The Local Coherence runtime is a CPU library, not a network service. It runs
in the address space of whatever application embeds it. The runtime never
opens sockets, spawns threads beyond `lc_set_threads`, or writes to disk on
its own. All I/O is the embedding application's responsibility.

This shapes the threat surface:

- **Untrusted inputs to the runtime are uint16 values + indices.** The
  injection / step / observation API takes integers and pointers. The
  runtime validates indices against tissue bounds; the application
  validates everything else.
- **No network or file parsers ship with the public runtime.** Any WAV /
  protocol / log parsing lives in the application layer.

## Hardening done in the 2026-05-07 review

- **`alloc_tissue` overflow check** — multiplications guarded against
  `size_t` overflow before `posix_memalign` / `VirtualAlloc`.
- **Strict-aliasing UB removed** from CPUID reads (memcpy into
  `uint32_t`, not pointer cast).
- **Index bounds validation** on `lc_inject_1d` / `lc_inject_2d` /
  `lc_get_field`.
- **Bit-exact regression suite** runs on every change to CORE files
  (see [`docs/architecture.md`](docs/architecture.md)). A change that
  produces a different golden number is rejected as a regression — this
  is enforced by the test harness, not by social convention.

## What has changed since the original review

- **ABI 1.2**: `lc_set_kernel` exposed in the public C API plus a
  `kernel_id` field on the tissue (`CanonicalKernel`, `SimpleAvgKernel`,
  `EmaKernel`, `CanonicalKernel_u64`).
- **`HotField<T>` template**: the cell type can now be `uint8_t`,
  `uint16_t`, or `uint32_t`. The original audit only covered `uint16_t`.
- Internal refactors and a flatten of the source layout. CORE file
  contents are largely the same but the build surface has shifted.

These changes have not been re-audited. Treat any v1.1+ code (pluggable
kernel, templated tissue) as carrying the same threat model as v1.0 by
inheritance, but not as having an explicit fresh attestation.

## Out of scope

- Production-grade hardening of the audio frontend or any application
  example. Examples in `examples/` and `apps/` are for demonstration;
  any production embedding of these patterns is the embedder's
  responsibility to harden.
- Side-channel resistance (timing, cache, power). The runtime is
  deterministic but not constant-time; do not use for cryptographic
  workloads.
- Privilege escalation. `lc_create_1d` will try `VirtualLock` /
  `mlock` for large pages on platforms that allow it, but falls back
  to ordinary pages on failure — no privilege drop, no setuid path.

## Reporting

If you find a security issue, please open a GitHub issue (or email the
maintainer directly for embargoed disclosure). There is no
vulnerability-disclosure program at this stage, but responsible reports
are welcome.

## Known limitations

- The runtime does not validate the integrity of a buffer passed to
  `lc_create_from_buffer`. Garbage-in, garbage-out — but it will not
  crash or read out of bounds within the buffer if `n` is correct.
- Multi-thread tissue access is **not** safe. Shard at the application
  level or serialize externally.
