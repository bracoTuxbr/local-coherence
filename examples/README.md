# Examples

Short C/C++ demos that exercise the `liblc` runtime directly. None are toys —
each corresponds to a result reported in the paper or an experiment from the
benchmark suite.

| File | What it shows |
|---|---|
| [`01_minimal_1d.cpp`](01_minimal_1d.cpp) | Manhattan-diamond growth `\|A(t)\| = 2t+1` predicted by PAD |
| [`pulse_1d.c`](pulse_1d.c) | Single-pulse propagation in 1D, observing `r_eff(t)` over time |
| [`anomaly_1d.c`](anomaly_1d.c) | Sliding-window anomaly detector on a synthetic time series |
| [`mel_2d.c`](mel_2d.c) | 2D tissue, propagation for 50 generations, dumps final field |

## Python users

Python examples (NAB anomaly, VAD, KWS, HAI, plus a Python translation of
the freeze speedup demo) live in the separate
[`lcruntime-python`](https://github.com/bracoTuxbr/lcruntime-python) repo,
under `examples/`.

## Building the C/C++ examples

You need the built shared library (`liblc.dll` on Windows, `liblc.so` on
Linux). From the repository root:

```bash
./tools/build.ps1                 # Windows MinGW
# or
g++ -O3 -mavx2 -mfma -std=c++17 src/lc.cpp -shared -fPIC -o build/liblc.so
```

Then build any example:

```bash
g++ -O2 -std=c++17 examples/01_minimal_1d.cpp -Isrc -o min1d
./min1d

gcc -O2 -std=c99 examples/pulse_1d.c -Iinclude -Lbuild -llc -o pulse
./pulse
```
