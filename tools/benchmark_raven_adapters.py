#!/usr/bin/env python3
"""Benchmark Raven's persistent HTTP adapter and optional direct C FFI."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import statistics
import sys
import tempfile
import time
import wave
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from audio import AudioError, inspect_wav  # noqa: E402
from voice_providers import VoiceRequest  # noqa: E402
from voice_providers.pocket_tts_raven import (  # noqa: E402
    PocketTtsRavenProvider,
    load_raven_installation,
)


SETTINGS = {
    "precision": "int8",
    "temperature": 0.2,
    "lsd_steps": 4,
    "threads": 0,
    "soften_commas": True,
}
DEFAULT_TEXT = "The signal is clean. We can begin the final approach now."
SAMPLE_RATE = 24_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", type=Path, required=True, help="consented WAV or MP3 reference")
    parser.add_argument("--voice-id", default="benchmark-voice")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--iterations", type=int, default=4, help="requests per adapter, minimum 2")
    parser.add_argument("--ffi-library", type=Path, help="optional libpocket_tts shared library")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/raven-adapter-benchmark/results.json"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(adapter: str, startup: float, requests: list[dict[str, float]]) -> dict[str, object]:
    warm = requests[1:]
    return {
        "adapter": adapter,
        "startup_seconds": startup,
        "requests": requests,
        "first_request_seconds": requests[0]["elapsed_seconds"],
        "warm_median_seconds": statistics.median(item["elapsed_seconds"] for item in warm),
        "warm_median_real_time_factor": statistics.median(item["real_time_factor"] for item in warm),
    }


def benchmark_server(
    voice: Path,
    voice_id: str,
    text: str,
    iterations: int,
    output: Path,
) -> dict[str, object]:
    installation = load_raven_installation()
    voices = output / "server-voices"
    provider = PocketTtsRavenProvider(
        installation=installation,
        voices_dir=voices,
        settings=SETTINGS,
    )
    provider.stage_voice(voice_id, voice)
    started = time.monotonic()
    provider.__enter__()
    startup = time.monotonic() - started
    requests = []
    try:
        for index in range(iterations):
            destination = output / "server" / f"request-{index + 1}.wav"
            started = time.monotonic()
            provider.synthesize(VoiceRequest(
                text=text,
                voice_id=voice_id,
                model="pocket-tts-raven",
                settings=SETTINGS,
                output_path=destination,
                idempotency_key=f"server-{index}",
            ))
            elapsed = time.monotonic() - started
            duration = inspect_wav(destination).duration_seconds
            requests.append({
                "elapsed_seconds": elapsed,
                "audio_seconds": duration,
                "real_time_factor": elapsed / duration,
            })
    finally:
        provider.close()
    return summarize("persistent-http", startup, requests)


def configure_ffi(library: ctypes.CDLL) -> None:
    library.ptt_create.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_float, ctypes.c_int, ctypes.c_int,
    ]
    library.ptt_create.restype = ctypes.c_void_p
    library.ptt_destroy.argtypes = [ctypes.c_void_p]
    library.ptt_set_soften_commas.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.ptt_stream_start.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    library.ptt_stream_start.restype = ctypes.c_void_p
    library.ptt_stream_read.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_int),
    ]
    library.ptt_stream_read.restype = ctypes.c_int
    library.ptt_stream_end.argtypes = [ctypes.c_void_p]
    library.ptt_free_audio.argtypes = [ctypes.POINTER(ctypes.c_float)]


def synthesize_ffi(library: ctypes.CDLL, handle: int, text: str, voice: str, output: Path) -> float:
    stream = library.ptt_stream_start(handle, text.encode(), voice.encode())
    if not stream:
        raise RuntimeError("ptt_stream_start failed")
    samples: list[float] = []
    try:
        while True:
            chunk = ctypes.POINTER(ctypes.c_float)()
            length = ctypes.c_int()
            status = library.ptt_stream_read(stream, ctypes.byref(chunk), ctypes.byref(length))
            if status == 0:
                break
            if status != 1:
                raise RuntimeError(f"ptt_stream_read returned {status}")
            try:
                samples.extend(chunk[index] for index in range(length.value))
            finally:
                library.ptt_free_audio(chunk)
    finally:
        library.ptt_stream_end(stream)

    output.parent.mkdir(parents=True, exist_ok=True)
    pcm = bytearray()
    for sample in samples:
        value = round(max(-1.0, min(1.0, sample)) * 32767)
        pcm.extend(int(value).to_bytes(2, "little", signed=True))
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return len(samples) / SAMPLE_RATE


def benchmark_ffi(
    library_path: Path,
    voice: Path,
    voice_id: str,
    text: str,
    iterations: int,
    output: Path,
) -> dict[str, object]:
    installation = load_raven_installation()
    voices = output / "ffi-voices"
    stager = PocketTtsRavenProvider(installation=installation, voices_dir=voices)
    staged_voice = stager.stage_voice(voice_id, voice)
    library = ctypes.CDLL(str(library_path))
    configure_ffi(library)
    started = time.monotonic()
    handle = library.ptt_create(
        str(installation.models).encode(),
        str(voices).encode(),
        str(installation.models / "tokenizer.model").encode(),
        str(SETTINGS["precision"]).encode(),
        ctypes.c_float(SETTINGS["temperature"]),
        SETTINGS["lsd_steps"],
        SETTINGS["threads"],
    )
    startup = time.monotonic() - started
    if not handle:
        raise RuntimeError("ptt_create failed")
    library.ptt_set_soften_commas(handle, 1)
    requests = []
    try:
        for index in range(iterations):
            destination = output / "ffi" / f"request-{index + 1}.wav"
            started = time.monotonic()
            duration = synthesize_ffi(library, handle, text, staged_voice.name, destination)
            elapsed = time.monotonic() - started
            requests.append({
                "elapsed_seconds": elapsed,
                "audio_seconds": duration,
                "real_time_factor": elapsed / duration,
            })
    finally:
        library.ptt_destroy(handle)
    result = summarize("ctypes", startup, requests)
    result["library"] = str(library_path)
    result["library_sha256"] = sha256_file(library_path)
    return result


def main() -> int:
    args = parse_args()
    voice = args.voice.expanduser().resolve()
    if not voice.is_file():
        raise ValueError(f"voice reference does not exist: {voice}")
    if args.iterations < 2:
        raise ValueError("--iterations must be at least 2")
    output_file = args.output.expanduser().resolve()
    output = output_file.parent
    output.mkdir(parents=True, exist_ok=True)

    installation = load_raven_installation()
    results: dict[str, object] = {
        "schema_version": 1,
        "platform": platform.platform(),
        "python": sys.version,
        "raven_revision": installation.raven_revision,
        "installation_manifest_sha256": sha256_file(installation.manifest_path),
        "voice_reference_sha256": sha256_file(voice),
        "text": args.text,
        "settings": SETTINGS,
        "iterations": args.iterations,
        "adapters": [benchmark_server(voice, args.voice_id, args.text, args.iterations, output)],
    }
    if args.ffi_library:
        library = args.ffi_library.expanduser().resolve()
        if not library.is_file():
            raise ValueError(f"FFI library does not exist: {library}")
        results["adapters"].append(
            benchmark_ffi(library, voice, args.voice_id, args.text, args.iterations, output)
        )
    output_file.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Results: {output_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, AudioError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
