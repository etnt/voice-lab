#!/usr/bin/env python3
"""Generate one Pocket TTS Raven voice line with the managed local runtime."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from audio import (
    AudioError,
    canonical_synthesis_fingerprint,
    sha256_file,
)
from voice_providers import VoiceRequest
from voice_providers.pocket_tts_raven import (
    PocketTtsRavenProvider,
    load_raven_installation,
)


VOICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one dialogue WAV with Pocket TTS Raven."
    )
    parser.add_argument("--voice", required=True, type=Path,
                        help="consented WAV or MP3 voice reference")
    parser.add_argument("--text", required=True, help="text to synthesize")
    parser.add_argument("--output", required=True, type=Path,
                        help="destination PCM16 WAV")
    parser.add_argument("--voice-id",
                        help="stable native voice ID (default: reference stem)")
    parser.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--lsd-steps", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--keep-commas", action="store_true",
                        help="preserve comma, semicolon, and colon pauses")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reference = args.voice.expanduser().resolve()
    if not reference.is_file() or reference.suffix.lower() not in {".wav", ".mp3"}:
        raise ValueError("--voice must be an existing WAV or MP3 file")
    if not args.text.strip():
        raise ValueError("--text must not be empty")
    if args.lsd_steps < 1 or args.threads < 0:
        raise ValueError("--lsd-steps must be positive and --threads non-negative")
    voice_id = args.voice_id or reference.stem
    if not VOICE_ID_PATTERN.fullmatch(voice_id):
        raise ValueError(
            "--voice-id must use filename-safe letters, digits, '.', '_' or '-'"
        )

    installation = load_raven_installation()
    settings = {
        "precision": args.precision,
        "temperature": args.temperature,
        "lsd_steps": args.lsd_steps,
        "threads": args.threads,
        "soften_commas": not args.keep_commas,
    }
    voices_dir = args.output.expanduser().resolve().parent / ".raven-cache"
    provider = PocketTtsRavenProvider(
        installation=installation,
        voices_dir=voices_dir,
        settings=settings,
    )
    if provider.base_url is None:
        provider.stage_voice(voice_id, reference)
    fingerprint = canonical_synthesis_fingerprint(
        {
            "text": args.text,
            "voice_id": voice_id,
            "reference_audio_sha256": sha256_file(reference),
            "engine_revision": installation.raven_revision,
            "onnx_runtime_version": installation.onnx_runtime_version,
            "model_manifest_sha256": installation.model_manifest_sha256,
            "settings": settings,
        }
    )
    request = VoiceRequest(
        text=args.text,
        voice_id=voice_id,
        model="pocket-tts-raven",
        settings=settings,
        output_path=args.output.expanduser().resolve(),
        idempotency_key=fingerprint,
    )
    with provider:
        result = provider.synthesize(request)
    print(f"Generated voice WAV: {result.output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, AudioError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
