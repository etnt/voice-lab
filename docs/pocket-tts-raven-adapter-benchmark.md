# Pocket TTS Raven Adapter Benchmark

Measured: 2026-09-08

## Decision

Keep the persistent loopback HTTP server as the production adapter. Direct
`ctypes` FFI did not improve warm synthesis latency and would move native crashes
into the storyboard runner process while adding another binary artifact and ABI
to package.

The FFI path remains an opt-in benchmark in
`tools/benchmark_raven_adapters.py`; it is not part of the supported runtime
installation.

## Method

The comparison used:

- macOS 26.6.2 on arm64.
- Python 3.14.7.
- Raven revision `abd26158ab50f954616eaf42296b09c4856489d7`.
- A project-author-owned Reginald Ashworth reference.
- The accepted production profile: int8, temperature 0.20, four LSD steps,
  automatic thread count, and comma softening.
- The same sentence and five requests per adapter.
- Independent, initially empty voice caches for the two adapters.
- Raven's upstream `BUILD_SHARED_LIB=ON` target for the disposable FFI build.

The result is stored locally at
`experiments/raven-adapter-benchmark/fresh-comparison/results.json`.

## Results

| Measurement | Persistent HTTP | Direct `ctypes` | FFI difference |
| --- | ---: | ---: | ---: |
| Engine/server startup | 0.402 s | 0.311 s | 0.091 s faster |
| First request, including voice encoding | 1.203 s | 1.143 s | 0.060 s faster |
| Warm request median | 0.558 s | 0.579 s | 0.021 s slower |
| Warm real-time-factor median | 0.176 | 0.163 | Both faster than real time |

Generated duration varies because synthesis is stochastic, so real-time factor
is useful for capacity context but not a pure transport-overhead measurement.
Wall-clock warm latency is the deciding metric here. This is a single-machine,
single-voice microbenchmark, not a general Raven performance claim.

## Reproduction

Build the optional shared-library target from the pinned Raven submodule into a
temporary or user-cache directory, keeping its ONNX Runtime dylib beside it.
Then run:

```bash
python3 tools/benchmark_raven_adapters.py \
  --voice /path/to/consented-reference.wav \
  --voice-id benchmark-voice \
  --iterations 5 \
  --ffi-library /path/to/libpocket_tts.dylib \
  --output experiments/raven-adapter-benchmark/results.json
```

Omit `--ffi-library` to measure only the supported persistent HTTP adapter.
Benchmark outputs and voice caches are local experiment artifacts and must not
be treated as distributable voice assets.
