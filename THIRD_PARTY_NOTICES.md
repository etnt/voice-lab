# Third-party notices

This project pins Pocket TTS Raven as source under
`vendor/pocket-tts-raven`. Raven is licensed under the MIT License; its license
text is retained at `vendor/pocket-tts-raven/LICENSE`.

Raven's own `vendor/pocket-tts-raven/THIRD_PARTY_NOTICES.md` is the
authoritative inventory for its vendored and build-time dependencies, including
ONNX Runtime, SentencePiece, and dr_libs. Those components remain under their
respective licenses.

The Raven setup obtains the `english_2026-04` ONNX bundle from
`KevinAHM/pocket-tts-onnx`, derived from `kyutai/pocket-tts`. Both model cards
declare the model weights under CC BY 4.0; the export repository declares its
code under Apache 2.0. Setup downloads hash-pinned inputs only after explicit
acceptance, then applies deterministic graph transformations locally.

Pocket TTS weights, transformed ONNX files, prebuilt Raven binaries, runtime
libraries, and voice references are not distributed by this repository. The
upstream model terms prohibit unauthorized voice impersonation or cloning,
deception and fraud, presenting generated content as a genuine recording, and
unlawful, harmful, or privacy-invasive use. A user-supplied voice reference must
have an explicit lawful consent basis; upstream sample voices have separate
licenses that must be reviewed before use.

Local installation manifests contain SHA-256 checksums but no publisher
signature. They detect local corruption and do not authenticate a release.
See `docs/pocket-tts-raven-distribution-review.md` for exact provenance,
attribution, the supported-platform boundary, and the prebuilt-release gate.