// make_test_audio.cpp — generates a realistic composite WAV to test the VAD
//
// Structure: JFK + silence + noise + JFK + silence.
// JFK is read from jfk.wav and concatenated.

#include "wav.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

using namespace cl;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: make_test_audio <jfk_in.wav> <out.wav>\n");
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];

    WavData jfk;
    if (!wav_read_pcm16(in_path, jfk)) {
        std::fprintf(stderr, "ERROR: failed to read %s\n", in_path.c_str());
        return 1;
    }
    std::fprintf(stderr, "JFK: %.2fs (%d samples)\n",
        jfk.samples.size() / (double)jfk.sample_rate,
        (int)jfk.samples.size());

    WavData out;
    out.sample_rate = jfk.sample_rate;
    out.n_channels = 1;

    // realistic scenario: Instagram video with little dialogue
    // 5s initial silence
    gen_silence(out, 5.0, 0.001);
    // first JFK (11s)
    out.samples.insert(out.samples.end(), jfk.samples.begin(), jfk.samples.end());
    // 15s silence (symbolic music = silence for this simulation)
    gen_silence(out, 15.0, 0.001);
    // 5s light noise (ambient)
    gen_noise(out, 5.0, 0.05);
    // 8s silence
    gen_silence(out, 8.0, 0.001);
    // second JFK (11s)
    out.samples.insert(out.samples.end(), jfk.samples.begin(), jfk.samples.end());
    // 12s silence
    gen_silence(out, 12.0, 0.001);
    // 4s noise
    gen_noise(out, 4.0, 0.05);
    // 6s silence
    gen_silence(out, 6.0, 0.001);
    // third JFK
    out.samples.insert(out.samples.end(), jfk.samples.begin(), jfk.samples.end());
    // 10s final silence
    gen_silence(out, 10.0, 0.001);

    if (!wav_write_pcm16(out_path, out)) {
        std::fprintf(stderr, "ERROR: failed to write %s\n", out_path.c_str());
        return 1;
    }

    double total_sec = out.samples.size() / (double)out.sample_rate;
    std::fprintf(stderr, "OK: %s — %.2fs total (%d samples)\n",
        out_path.c_str(), total_sec, (int)out.samples.size());

    double jfk_sec = jfk.samples.size() / (double)jfk.sample_rate;
    double speech_total = 3 * jfk_sec;
    double silence_total = 5 + 15 + 8 + 12 + 6 + 10;
    double noise_total   = 5 + 4;
    std::fprintf(stderr, "breakdown:\n");
    std::fprintf(stderr, "  speech (JFK x3)     : %.2fs (%.1f%%)\n", speech_total, 100*speech_total/total_sec);
    std::fprintf(stderr, "  silence (6 parts)   : %.2fs (%.1f%%)\n", silence_total, 100*silence_total/total_sec);
    std::fprintf(stderr, "  light noise (2x)    : %.2fs (%.1f%%)\n", noise_total, 100*noise_total/total_sec);

    return 0;
}
