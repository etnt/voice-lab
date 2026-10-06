#!/usr/bin/env python3
"""Synthesize a JSON manuscript of voiced lines with Pocket TTS Raven.

The manuscript JSON declares named voices (each pointing at a consented
WAV/MP3 reference) and an ordered list of lines that cite those voices.
Lines are grouped by effective synthesis settings so each unique settings
combination runs on one persistent Raven server session.

See .agents/skills/manuscript-voice/SKILL.md for the full JSON format.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audio import (
    AudioError,
    canonical_synthesis_fingerprint,
    normalize_tts_text,
    sha256_file,
)
from voice_providers import VoiceRequest
from voice_providers.pocket_tts_raven import (
    PocketTtsRavenProvider,
    load_raven_installation,
)

MANUSCRIPT_VERSION = 1
VOICE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
REFERENCE_SUFFIXES = {".wav", ".mp3"}
ALLOWED_OVERRIDES = (
    "precision",
    "temperature",
    "lsd_steps",
    "threads",
    "keep_commas",
)


class ManuscriptError(ValueError):
    """Raised when a manuscript file is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class VoiceDefinition:
    name: str
    reference: Path
    voice_id: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManuscriptLine:
    index: int
    voice_name: str
    text: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Manuscript:
    path: Path
    title: str
    output_dir: Path
    voices: dict[str, VoiceDefinition]
    settings: dict[str, Any]
    lines: list[ManuscriptLine]

    def effective_settings(self, line: ManuscriptLine) -> dict[str, Any]:
        voice = self.voices[line.voice_name]
        merged = {**self.settings, **voice.overrides, **line.overrides}
        unknown = set(merged) - set(ALLOWED_OVERRIDES)
        if unknown:
            raise ManuscriptError(
                f"line {line.index}: unknown setting key(s) {sorted(unknown)}; "
                f"allowed: {sorted(ALLOWED_OVERRIDES)}"
            )
        return merged

    def settings_groups(self) -> list[tuple[tuple, list[tuple[ManuscriptLine, dict[str, Any]]]]]:
        """Group lines by effective settings, preserving first-appearance order."""
        groups: dict[tuple, list[tuple[ManuscriptLine, dict[str, Any]]]] = {}
        for line in self.lines:
            settings = self.effective_settings(line)
            key = tuple(sorted(settings.items()))
            groups.setdefault(key, []).append((line, settings))
        return list(groups.items())


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ManuscriptError(message)


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    _expect(isinstance(value, dict), f"{location} must be a JSON object")
    return value


def _require_str(value: Any, location: str) -> str:
    _expect(isinstance(value, str) and value.strip(), f"{location} must be a non-empty string")
    return value.strip()


def load_manuscript(path: Path) -> Manuscript:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ManuscriptError(f"manuscript file not found: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManuscriptError(f"{path} is not valid JSON: {exc}") from exc
    document = _require_mapping(document, "manuscript root")

    version = document.get("version", MANUSCRIPT_VERSION)
    _expect(
        version == MANUSCRIPT_VERSION,
        f"unsupported manuscript version {version!r}; expected {MANUSCRIPT_VERSION}",
    )

    voices_document = _require_mapping(
        document.get("voices"), "voices"
    )
    _expect(bool(voices_document), "voices must declare at least one voice")
    voices: dict[str, VoiceDefinition] = {}
    for name, spec in voices_document.items():
        _expect(bool(_require_str(name, "voice name")), "voice name must not be empty")
        _expect(
            re.fullmatch(VOICE_ID_PATTERN, name) is not None,
            f"voice name {name!r} must use filename-safe letters, digits, '.', '_' or '-'",
        )
        spec = _require_mapping(spec, f"voices.{name}")
        reference_text = _require_str(spec.get("reference"), f"voices.{name}.reference")
        reference = (path.parent / reference_text).resolve()
        _expect(
            reference.is_file() and reference.suffix.lower() in REFERENCE_SUFFIXES,
            f"voices.{name}.reference must be an existing WAV or MP3 file, got {reference}",
        )
        voice_id = spec.get("voice_id", name)
        voice_id = _require_str(voice_id, f"voices.{name}.voice_id")
        _expect(
            re.fullmatch(VOICE_ID_PATTERN, voice_id) is not None,
            f"voices.{name}.voice_id {voice_id!r} must use filename-safe letters, "
            "digits, '.', '_' or '-'",
        )
        overrides = {
            key: value
            for key, value in spec.items()
            if key not in {"reference", "voice_id"}
        }
        voices[name] = VoiceDefinition(
            name=name, reference=reference, voice_id=voice_id, overrides=overrides
        )

    lines_document = document.get("lines")
    _expect(isinstance(lines_document, list), "lines must be a JSON array")
    _expect(bool(lines_document), "lines must contain at least one line")
    lines: list[ManuscriptLine] = []
    for position, entry in enumerate(lines_document, start=1):
        entry = _require_mapping(entry, f"lines[{position}]")
        voice_name = _require_str(entry.get("voice"), f"lines[{position}].voice")
        _expect(
            voice_name in voices,
            f"lines[{position}].voice {voice_name!r} is not declared in voices "
            f"(declared: {sorted(voices)})",
        )
        text = _require_str(entry.get("text"), f"lines[{position}].text")
        overrides = {
            key: value for key, value in entry.items() if key not in {"voice", "text"}
        }
        lines.append(
            ManuscriptLine(
                index=position, voice_name=voice_name, text=text, overrides=overrides
            )
        )

    settings = document.get("settings", {})
    settings = {
        key: value
        for key, value in _require_mapping(settings, "settings").items()
    }
    _expect(
        set(settings) <= set(ALLOWED_OVERRIDES),
        f"settings has unknown key(s) {sorted(set(settings) - set(ALLOWED_OVERRIDES))}; "
        f"allowed: {sorted(ALLOWED_OVERRIDES)}",
    )

    output_dir_text = document.get("output_dir", "out")
    output_dir = (path.parent / _require_str(output_dir_text, "output_dir")).resolve()

    return Manuscript(
        path=path,
        title=document.get("title", path.stem),
        output_dir=output_dir,
        voices=voices,
        settings=settings,
        lines=lines,
    )


def convert_to_mp3(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source), "-codec:a", "libmp3lame", "-qscale:a", "2",
            str(destination),
        ],
        check=True,
    )


def concatenate_mp3s(parts: list[Path], destination: Path) -> None:
    concat_list = destination.parent / f".{destination.stem}-concat.txt"
    with concat_list.open("w", encoding="utf-8") as stream:
        for part in parts:
            stream.write(f"file '{part.resolve()}'\n")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-codec:a", "libmp3lame", "-qscale:a", "2", str(destination),
            ],
            check=True,
        )
    finally:
        concat_list.unlink(missing_ok=True)


def synthesize_manuscript(
    manuscript: Manuscript,
    *,
    out_dir: Path,
    keep_wav: bool,
    concat_path: Path | None,
    dry_run: bool,
) -> list[Path]:
    installation = load_raven_installation()
    out_dir.mkdir(parents=True, exist_ok=True)
    voices_dir = out_dir / ".raven-cache"
    voices_dir.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []
    open_servers: list[PocketTtsRavenProvider] = []
    try:
        for _, group in manuscript.settings_groups():
            settings = group[0][1]
            provider_settings = {
                **settings,
                "soften_commas": not settings.get("keep_commas", False),
            }
            provider_settings.pop("keep_commas", None)
            provider = PocketTtsRavenProvider(
                installation=installation,
                voices_dir=voices_dir,
                settings=provider_settings,
            )
            open_servers.append(provider)
            if dry_run:
                for line, _ in group:
                    voice = manuscript.voices[line.voice_name]
                    print(
                        f"[dry-run] {line.index:03d} [{voice.name}] "
                        f"settings={provider_settings} text={line.text!r}"
                    )
                    produced.append(
                        out_dir / f"{line.index:03d}-{voice.voice_id}.mp3"
                    )
                continue

            staged: set[str] = set()
            for line, _ in group:
                voice = manuscript.voices[line.voice_name]
                if voice.voice_id not in staged:
                    _expect(
                        re.fullmatch(VOICE_ID_PATTERN, voice.voice_id) is not None,
                        f"voice id {voice.voice_id!r} is not filename-safe",
                    )
                    provider.stage_voice(voice.voice_id, voice.reference)
                    staged.add(voice.voice_id)

            with provider:
                for line, _ in group:
                    voice = manuscript.voices[line.voice_name]
                    text = normalize_tts_text(line.text)
                    fingerprint = canonical_synthesis_fingerprint(
                        {
                            "text": text,
                            "voice_id": voice.voice_id,
                            "reference_audio_sha256": sha256_file(voice.reference),
                            "engine_revision": installation.raven_revision,
                            "onnx_runtime_version": installation.onnx_runtime_version,
                            "model_manifest_sha256": installation.model_manifest_sha256,
                            "settings": provider_settings,
                        }
                    )
                    wav_path = out_dir / f"{line.index:03d}-{voice.voice_id}.wav"
                    request = VoiceRequest(
                        text=text,
                        voice_id=voice.voice_id,
                        model="pocket-tts-raven",
                        settings=provider_settings,
                        output_path=wav_path,
                        idempotency_key=fingerprint,
                    )
                    provider.synthesize(request)
                    mp3_path = wav_path.with_suffix(".mp3")
                    convert_to_mp3(wav_path, mp3_path)
                    if not keep_wav:
                        wav_path.unlink(missing_ok=True)
                    print(f"{line.index:03d} [{voice.name}] {mp3_path}")
                    produced.append(mp3_path)
    finally:
        for provider in open_servers:
            provider.close()

    if concat_path is not None and not dry_run and produced:
        concatenate_mp3s(produced, concat_path)
        print(f"Concatenated manuscript audio: {concat_path}")
    return produced


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize a JSON manuscript of voiced lines with "
                    "Pocket TTS Raven."
    )
    parser.add_argument("manuscript", type=Path,
                        help="path to the manuscript JSON file")
    parser.add_argument("--out-dir", type=Path,
                        help="destination directory (default: the manuscript's "
                             "output_dir, else out/ next to the manuscript)")
    parser.add_argument("--concat", type=Path,
                        help="also join all line MP3s into one file "
                             "(e.g. out/manuscript.mp3)")
    parser.add_argument("--keep-wav", action="store_true",
                        help="keep the intermediate PCM16 WAV files")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the manuscript and print the plan "
                             "without synthesizing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manuscript = load_manuscript(args.manuscript)
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else manuscript.output_dir
    )
    concat_path = (
        args.concat.expanduser().resolve() if args.concat is not None else None
    )
    produced = synthesize_manuscript(
        manuscript,
        out_dir=out_dir,
        keep_wav=args.keep_wav,
        concat_path=concat_path,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"Done: {len(produced)} line(s) from {manuscript.path.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ManuscriptError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
        AudioError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
