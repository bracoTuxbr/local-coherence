// extract_segments.cpp — extracts speech segments from a WAV into another WAV
//
// Reads segments.txt (lines "start_ms end_ms class") and the original WAV.
// Concatenates only the segments with class=1 into an output WAV.

#include "wav.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace cl;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: extract_segments <wav_in> <segments.txt> <wav_out>\n");
        return 1;
    }
    std::string wav_in = argv[1];
    std::string segs   = argv[2];
    std::string wav_out = argv[3];

    WavData in;
    if (!wav_read_pcm16(wav_in, in)) {
        std::fprintf(stderr, "ERROR: failed to read %s\n", wav_in.c_str());
        return 1;
    }
    int sr = in.sample_rate;

    std::ifstream fs(segs);
    if (!fs.is_open()) {
        std::fprintf(stderr, "ERROR: failed to open %s\n", segs.c_str());
        return 1;
    }

    WavData out;
    out.sample_rate = sr;
    out.n_channels = 1;

    int n_speech = 0;
    int total_in_ms = 0, total_out_ms = 0;
    std::string line;
    while (std::getline(fs, line)) {
        std::istringstream iss(line);
        int start_ms, end_ms, cls;
        if (!(iss >> start_ms >> end_ms >> cls)) continue;
        total_in_ms += (end_ms - start_ms);
        if (cls != 1) continue;
        size_t s_lo = (size_t)((double)start_ms * sr / 1000.0);
        size_t s_hi = (size_t)((double)end_ms   * sr / 1000.0);
        if (s_lo > in.samples.size()) s_lo = in.samples.size();
        if (s_hi > in.samples.size()) s_hi = in.samples.size();
        out.samples.insert(out.samples.end(),
            in.samples.begin() + s_lo, in.samples.begin() + s_hi);
        ++n_speech;
        total_out_ms += (end_ms - start_ms);
    }

    if (!wav_write_pcm16(wav_out, out)) {
        std::fprintf(stderr, "ERROR: failed to write %s\n", wav_out.c_str());
        return 1;
    }

    std::fprintf(stderr, "OK: %d speech segments extracted\n", n_speech);
    std::fprintf(stderr, "    input  total: %d ms (%.2fs)\n", total_in_ms, total_in_ms / 1000.0);
    std::fprintf(stderr, "    output speech: %d ms (%.2fs)\n", total_out_ms, total_out_ms / 1000.0);
    std::fprintf(stderr, "    reduction: %.1f%%\n",
        100.0 * (total_in_ms - total_out_ms) / (double)total_in_ms);
    return 0;
}
