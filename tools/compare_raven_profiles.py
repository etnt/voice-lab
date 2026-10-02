#!/usr/bin/env python3
"""Generate a blinded Pocket TTS Raven profile-comparison listening set."""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from audio import (
    AudioErrorCode,
    AudioError,
    ValidationIssue,
    inspect_wav,
    normalize_tts_text,
    plausible_audio_seconds,
    postprocess_dialogue_wav,
)
from voice_providers import VoiceRequest
from voice_providers.pocket_tts_raven import (
    PocketTtsRavenProvider,
    load_raven_installation,
)


CORPUS = (
    ("short", "Wait. Listen."),
    ("punctuation", '"Kerk," he said, "is the signal clean; or are we too late?"'),
    ("story", "Jason dinAlt crossed the landing bay and checked the Pyrran instruments."),
    ("urgent", "Move now! Seal the hatch, cut the engines, and do not answer that signal."),
    ("long", "The repair was delicate, but the old machine answered each careful adjustment with a steadier rhythm, until the room finally fell quiet."),
)
PROFILES = {
    "A": {"precision": "int8", "temperature": 0.7, "lsd_steps": 1, "threads": 0, "soften_commas": True},
    "B": {"precision": "int8", "temperature": 0.2, "lsd_steps": 4, "threads": 0, "soften_commas": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voice",
        action="append",
        required=True,
        metavar="ID=PATH",
        help="consented reference voice; repeat for each character",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/raven-profile-comparison"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_voices(values: list[str]) -> dict[str, Path]:
    voices: dict[str, Path] = {}
    for value in values:
        voice_id, separator, raw_path = value.partition("=")
        path = Path(raw_path).expanduser().resolve()
        if not separator or not voice_id or not path.is_file():
            raise ValueError(f"invalid --voice {value!r}; expected ID=EXISTING_PATH")
        voices[voice_id] = path
    return voices


def render_index(output: Path, voices: dict[str, Path], records: list[dict[str, object]]) -> None:
    rows = []
    line_order = {line_id: index for index, (line_id, _) in enumerate(CORPUS)}
    ordered = sorted(
        records,
        key=lambda record: (
            str(record["voice_id"]),
            line_order[str(record["line_id"])],
            str(record["profile"]),
        ),
    )
    for record in ordered:
        audio_path = Path(str(record["output"])).relative_to(output)
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record['voice_id']))}</td>"
            f"<td>{html.escape(str(record['line_id']))}</td>"
            f"<td>{html.escape(str(record['text']))}</td>"
            f"<td>{html.escape(str(record['profile']))}</td>"
            f"<td><audio controls preload=\"none\" src=\"{html.escape(audio_path.as_posix())}\"></audio></td>"
            f"<td>{float(record['duration_seconds']):.2f}s</td>"
            f"<td>{float(record['request_seconds']):.2f}s</td>"
            "</tr>"
        )
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Raven profile comparison</title><style>
:root{color-scheme:dark;font-family:Georgia,serif;background:#081a2b;color:#edf4f7}body{max-width:1100px;margin:0 auto;padding:32px}h1{font-size:2rem}table{width:100%;border-collapse:collapse;background:#102a43}th,td{padding:12px;border:1px solid #31506b;text-align:left}th{background:#173f5f}audio{width:100%;min-width:260px}.note{color:#a9c5d8}
</style></head><body><h1>Raven profile comparison</h1>
<p class="note">Compare A and B without opening profile-key.json. Use headphones and score intelligibility, speaker similarity, artifacts, and performance.</p>
<table><thead><tr><th>Voice</th><th>Line</th><th>Text</th><th>Profile</th><th>Audio</th><th>Duration</th><th>Generation</th></tr></thead><tbody>
""" + "\n".join(rows) + "\n</tbody></table></body></html>\n"
    (output / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    args = parse_args()
    voices = parse_voices(args.voice)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    installation = load_raven_installation()
    records: list[dict[str, object]] = []
    previous_records: dict[tuple[str, str, str], dict[str, object]] = {}
    try:
        previous = json.loads((output / "results.json").read_text(encoding="utf-8"))
        previous_records = {
            (item["profile"], item["voice_id"], item["line_id"]): item
            for item in previous
            if isinstance(item, dict)
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass

    for label, settings in PROFILES.items():
        profile_root = output / label
        provider = PocketTtsRavenProvider(
            installation=installation,
            voices_dir=profile_root / ".raven-cache",
            settings=settings,
        )
        for voice_id, reference in voices.items():
            provider.stage_voice(voice_id, reference)
        with provider:
            for voice_id in voices:
                for line_id, source_text in CORPUS:
                    destination = profile_root / voice_id / f"{line_id}.wav"
                    if destination.is_file() and not args.force:
                        info = inspect_wav(destination)
                        record = {
                            "profile": label,
                            "voice_id": voice_id,
                            "line_id": line_id,
                            "text": source_text,
                            "output": str(destination),
                            "duration_seconds": info.duration_seconds,
                            "request_seconds": 0.0,
                            "real_time_factor": 0.0,
                            "reused": True,
                        }
                        prior = previous_records.get((label, voice_id, line_id), {})
                        for field in ("request_seconds", "real_time_factor", "provider_metadata"):
                            if field in prior:
                                record[field] = prior[field]
                        records.append(record)
                        continue
                    text = normalize_tts_text(source_text)
                    raw = destination.with_name(destination.stem + ".raw.wav")
                    started = time.monotonic()
                    result = provider.synthesize(VoiceRequest(
                        text=text,
                        voice_id=voice_id,
                        model="pocket-tts-raven",
                        settings=settings,
                        output_path=raw,
                        idempotency_key=f"comparison-{label}-{voice_id}-{line_id}",
                    ))
                    try:
                        raw_info = inspect_wav(raw)
                        if raw_info.duration_seconds > plausible_audio_seconds(text):
                            raise AudioError([ValidationIssue(
                                AudioErrorCode.OUTPUT_TOO_LONG,
                                f"{voice_id}/{line_id}",
                                "Raven produced implausibly long comparison audio",
                            )])
                        info = postprocess_dialogue_wav(raw, destination)
                    finally:
                        raw.unlink(missing_ok=True)
                    elapsed = time.monotonic() - started
                    records.append({
                        "profile": label,
                        "voice_id": voice_id,
                        "line_id": line_id,
                        "text": source_text,
                        "output": str(destination),
                        "duration_seconds": info.duration_seconds,
                        "request_seconds": elapsed,
                        "real_time_factor": elapsed / info.duration_seconds,
                        "provider_metadata": dict(result.provider_metadata),
                        "reused": False,
                    })

    (output / "results.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (output / "profile-key.json").write_text(json.dumps(PROFILES, indent=2) + "\n", encoding="utf-8")
    render_index(output, voices, records)
    print(f"Listening set: {output / 'index.html'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, AudioError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
