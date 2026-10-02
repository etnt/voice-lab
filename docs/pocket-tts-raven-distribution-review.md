# Pocket TTS Raven Distribution Review

Reviewed: 2026-09-08

This document is an engineering compliance inventory, not legal advice. Review
upstream terms again before publishing model files, native binaries, containers,
or installers that contain third-party artifacts.

## Decision

The supported distribution mode is **local build only**:

- This repository may distribute its installer, provider code, pinned Raven Git
  submodule reference, and documentation.
- The installer may download hash-pinned model inputs only after the user passes
  `--accept-model-terms`, transform them locally, and build Raven locally.
- This repository does not distribute Pocket TTS model weights, transformed ONNX
  files, prebuilt Raven binaries, ONNX Runtime libraries, or voice references.
- A generated installation manifest is checksummed but not publisher-signed. It
  detects local corruption or tampering; it does not authenticate a release.
- Publishing prebuilt artifacts requires a new review and a defined signing-key,
  verification, rotation, and revocation policy.

## Model Provenance

The installer uses the `english_2026-04` bundle from the community ONNX export:

- Exact source: `KevinAHM/pocket-tts-onnx`, path
  `onnx/english_2026-04` on Hugging Face.
- Base model: `kyutai/pocket-tts`.
- Model license declared by both cards: Creative Commons Attribution 4.0
  International (`CC BY 4.0`).
- Export code license declared by the mirror: Apache License 2.0.
- Input URLs and SHA-256 values are pinned in
  `vendor/pocket-tts-raven/tools/prepare_models.sh`.
- The setup process performs deterministic local graph rewrites. The resulting
  ONNX files remain derived model artifacts and are not committed or published.

CC BY 4.0 attribution must identify Kyutai's Pocket TTS model, link the license,
and indicate that the ONNX export and local graph transformations were used.
The Raven third-party notice contains the detailed attribution and dependency
inventory and is copied into every local installation.

The official model repository is gated and requires acceptance of its current
conditions. Its prohibited-use terms include unauthorized voice impersonation
or cloning; deceptive, fraudulent, or misleading use; presenting generated
content as genuine recordings; and unlawful, harmful, or privacy-invasive use.
The ONNX mirror repeats those restrictions. Setup's acceptance flag records a
local timestamp but does not replace review of the current upstream terms.

## Voice References

Voice reference files are user-supplied project assets, not Raven installation
artifacts. A reference must have an explicit lawful consent basis recorded in
the storyboard voice registry. Do not package upstream sample voices or a real
person's voice without separately verifying that voice asset's license and the
speaker's consent.

## Native Components

The local build currently pins:

| Component | Identity | License source |
| --- | --- | --- |
| Pocket TTS Raven | `abd26158ab50f954616eaf42296b09c4856489d7` | `vendor/pocket-tts-raven/LICENSE` |
| ONNX Runtime | `1.23.2`, archive hash pinned by Raven CMake | Raven third-party notice |
| SentencePiece | `v0.2.1` | Raven third-party notice |
| dr_libs | `cd99e2cca5ccb48f8d000f1cd844424dcdbbaf59` | Raven third-party notice |

Every installed executable, dynamic library, model file, and notice is listed
with its relative path, byte size, role, and SHA-256 in manifest schema 2. The
loader rejects missing, modified, path-escaping, or undeclared files. Environment
overrides remain a development escape hatch and cannot establish provenance for
external artifacts.

## Supported Platforms

`macOS arm64` and `Linux x86_64` are the platforms currently built and
validated by this project. On Linux the installer builds with the system GNU
C++ compiler (`g++`) and the CMake build fetches the hash-pinned
`onnxruntime-linux-x64` release; the Apple-only Accelerate/AMX conv custom ops
are not compiled, so Raven uses its portable decoder backend there. Raven has
upstream Windows build branches, but Windows is not supported here until this
installer, artifact inventory, native dependencies, and integration suite pass
on that host.

## Release Gate

Before distributing a prebuilt runtime or model bundle:

1. Re-review all current upstream licenses, gated conditions, and notices.
2. Decide explicitly whether transformed model redistribution is permitted and
   satisfy CC BY 4.0 attribution requirements.
3. Define a publisher signing system and public-key trust path.
4. Sign the release manifest and verify it before trusting artifact checksums.
5. Publish complete corresponding notices with the release.
6. Validate installation and real synthesis on every advertised platform.

Until all six checks are complete, the manifest must continue to declare
`distribution.policy = "local-build-only"` and
`distribution.publisher_signature = null`.
