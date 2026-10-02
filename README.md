# voice-lab

Local voice cloning and speech generation with Pocket TTS Raven.

`voice-lab` is the standalone home of the Pocket TTS Raven voice pipeline that
was previously embedded in the ai-video-generator project. It deals only with
cloning a voice from a consented reference recording and generating speech
audio with the managed local Raven runtime.

## Layout

- `voice_providers/pocket_tts_raven.py` — persistent loopback HTTP provider
  that starts, health-checks, and drives the local Raven runtime; validates
  manifests by path, size, and SHA-256; stages and conditions voice references.
- `voice_providers/__init__.py` — `VoiceRequest`/`VoiceResult`/`VoiceProvider`
  contracts.
- `audio.py` — WAV inspection, PCM16 conditioning/normalization, text
  normalization, and synthesis fingerprints.
- `generate_voice.py` — one-shot CLI: reference WAV/MP3 + text → PCM16 WAV.
- `make_sentences.py` — interactive batch helper: pick a voice, prompt for
  sentences, write numbered MP3 takes via ffmpeg.
- `voices/` — built-in in-house reference voices (Reginald Ashworth, Deja
  Thoris); see `voices/README.md`.
- `tools/setup_raven.py` — hash-pinned local builder/installer for Raven and
  its ONNX models (`--accept-model-terms` required).
- `tools/compare_raven_profiles.py`, `tools/benchmark_raven_adapters.py` —
  profile comparison and adapter benchmarking utilities.
- `tests/test_pocket_tts_raven.py` — provider, installer-validation, and
  adapter test suite.
- `vendor/pocket-tts-raven` — Raven source, pinned as a Git submodule.
- `docs/` — distribution review and adapter benchmark documentation.

## Setup

```sh
git submodule update --init --recursive
python3 tools/setup_raven.py --accept-model-terms
python3 tools/setup_raven.py --check-installation
```

The installer downloads hash-pinned model inputs only after the user accepts
the model terms, rebuilds the ONNX graph locally, builds Raven from source in
isolation, and publishes a schema-v2 installation under
`~/Library/Caches/voice-lab/pocket-tts-raven/` only after a successful server
health check.

Supported platforms: macOS arm64 and Linux x86_64.

## Generate one voice line

The two built-in reference voices live in `voices/`. For batch work, use the
interactive helper — pick a voice, then type sentences one at a time and get
`out/1-out.mp3`, `out/2-out.mp3`, ... (numbering continues across runs; the
intermediate PCM16 WAV is kept next to each MP3):

```sh
python3 make_sentences.py
python3 make_sentences.py --voice voices/deja-thoris.wav --out-dir takes/
```

The prompt shows the active voice (`[deja-thoris] sentence>`). Type a sentence
to synthesize it with that voice, or enter a voice number (e.g. `2`) to switch
speakers mid-discussion; an empty line quits.

For a single line directly:

```sh
python3 generate_voice.py \
  --voice ./voices/speaker.wav \
  --text "Hello from voice-lab." \
  --output ./out/speaker-hello.wav
```

Options: `--precision int8|fp32`, `--temperature`, `--lsd-steps`, `--threads`,
`--keep-commas`, `--voice-id`. The voice reference must be an existing WAV or
MP3 file with an explicit lawful consent basis; see
`docs/pocket-tts-raven-distribution-review.md` for model attribution, consent
requirements, and the prebuilt-release gate.

## Benchmarking

```sh
python3 tools/benchmark_raven_adapters.py \
  --voice ./voices/speaker.wav \
  --ffi-library /path/to/libpocket_tts.dylib
python3 tools/compare_raven_profiles.py \
  --voice speaker=./voices/speaker.wav
```

The persistent loopback HTTP adapter is the production default; direct `ctypes`
FFI remains an opt-in benchmark path.

## Tests

```sh
python3 -m unittest discover -s tests
```

## Licensing

Project code is MPL 2.0 (`LICENSE`). The Pocket TTS model is CC BY 4.0
(`kyutai/pocket-tts`, ONNX export `KevinAHM/pocket-tts-onnx`); third-party
dependency notices are in `THIRD_PARTY_NOTICES.md` and the Raven submodule.
Model files, transformed ONNX artifacts, prebuilt binaries, and voice
references are not distributed by this repository.
