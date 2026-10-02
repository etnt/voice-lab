#!/usr/bin/env python3
"""Interactive helper: pick a voice, then clone sentences to numbered MP3s."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from audio import AudioError, canonical_synthesis_fingerprint, sha256_file
from voice_providers import VoiceRequest
from voice_providers.pocket_tts_raven import (
    PocketTtsRavenProvider,
    load_raven_installation,
)


VOICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick a reference voice, then synthesize sentences as "
                    "1-out.mp3, 2-out.mp3, ... using Pocket TTS Raven."
    )
    parser.add_argument("--voice", type=Path,
                        help="skip voice selection: path to a WAV or MP3 reference")
    parser.add_argument("--voices-dir", type=Path, default=Path("voices"),
                        help="directory of reference voices to choose from "
                             "(default: voices/)")
    parser.add_argument("--out-dir", type=Path, default=Path("out"),
                        help="destination directory (default: out/)")
    parser.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--lsd-steps", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--keep-commas", action="store_true",
                        help="preserve comma, semicolon, and colon pauses")
    return parser.parse_args(argv)


def choose_voice(args: argparse.Namespace) -> Path:
    if args.voice is not None:
        reference = args.voice.expanduser().resolve()
        if not reference.is_file() or reference.suffix.lower() not in {".wav", ".mp3"}:
            raise ValueError("--voice must be an existing WAV or MP3 file")
        return reference

    references = sorted(
        path for path in args.voices_dir.expanduser().resolve().iterdir()
        if path.suffix.lower() in {".wav", ".mp3"}
    ) if args.voices_dir.expanduser().resolve().is_dir() else []
    if not references:
        raise ValueError(
            f"no WAV or MP3 references found in {args.voices_dir}; "
            "pass --voice explicitly"
        )
    print("Available voices:")
    for index, reference in enumerate(references, start=1):
        print(f"  {index}. {reference.stem}")
    while True:
        choice = input(f"Choose voice [1-{len(references)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(references):
            return references[int(choice) - 1]
        print("Please enter a number from the list.")


def next_sentence_index(out_dir: Path) -> int:
    highest = 0
    for path in out_dir.glob("*-out.mp3"):
        match = re.match(r"^(\d+)-out\.mp3$", path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def convert_to_mp3(source: Path, destination: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(source), "-codec:a", "libmp3lame", "-qscale:a", "2",
         str(destination)],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.lsd_steps < 1 or args.threads < 0:
        raise ValueError("--lsd-steps must be positive and --threads non-negative")

    reference = choose_voice(args)
    voice_id = reference.stem
    if not VOICE_ID_PATTERN.fullmatch(voice_id):
        raise ValueError(
            f"voice id {voice_id!r} must use filename-safe letters, digits, "
            "'.', '_' or '-'"
        )
    print(f"Voice: {reference} (id {voice_id!r})")

    installation = load_raven_installation()
    settings = {
        "precision": args.precision,
        "temperature": args.temperature,
        "lsd_steps": args.lsd_steps,
        "threads": args.threads,
        "soften_commas": not args.keep_commas,
    }
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    voices_dir = out_dir / ".raven-cache"
    provider = PocketTtsRavenProvider(
        installation=installation,
        voices_dir=voices_dir,
        settings=settings,
    )
    if provider.base_url is None:
        provider.stage_voice(voice_id, reference)

    index = next_sentence_index(out_dir)
    print("Enter sentences one at a time; an empty line quits.")
    with provider:
        while True:
            try:
                text = input("sentence> ").strip()
            except EOFError:
                break
            if not text:
                break
            fingerprint = canonical_synthesis_fingerprint(
                {
                    "text": text,
                    "voice_id": voice_id,
                    "reference_audio_sha256": sha256_file(reference),
                    "engine_revision": installation.raven_revision,
                    "onnx_runtime_version": installation.onnx_runtime_version,
                    "model_manifest_sha256": installation.model_manifest_sha256,
                    "settings": settings,
                }
            )
            wav_path = out_dir / f"{index}-out.wav"
            request = VoiceRequest(
                text=text,
                voice_id=voice_id,
                model="pocket-tts-raven",
                settings=settings,
                output_path=wav_path,
                idempotency_key=fingerprint,
            )
            result = provider.synthesize(request)
            mp3_path = out_dir / f"{index}-out.mp3"
            convert_to_mp3(wav_path, mp3_path)
            print(f"{mp3_path}")
            index += 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError, AudioError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
