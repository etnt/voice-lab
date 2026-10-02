"""Shared WAV inspection, conditioning, and synthesis-fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


AUDIO_POLICY_VERSION = 2


class AudioErrorCode(str, Enum):
    INVALID_WAV = "invalid_wav"
    VOICE_INVALID = "voice_invalid"
    RUNTIME_MISSING = "runtime_missing"
    MODEL_INVALID = "model_invalid"
    STARTUP_TIMEOUT = "startup_timeout"
    REQUEST_TIMEOUT = "request_timeout"
    SYNTHESIS_FAILURE = "synthesis_failure"
    SHUTDOWN_TIMEOUT = "shutdown_timeout"
    OUTPUT_TOO_LONG = "output_too_long"


@dataclass(frozen=True)
class ValidationIssue:
    code: AudioErrorCode
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message} [{self.code.value}]"


class AudioError(Exception):
    def __init__(self, issues: Sequence[ValidationIssue]):
        if not issues:
            raise ValueError("AudioError requires at least one issue")
        self.issues = tuple(issues)
        super().__init__("\n".join(str(issue) for issue in self.issues))


@dataclass(frozen=True)
class WavInfo:
    sample_format: str
    sample_rate_hz: int
    channels: int
    bits_per_sample: int
    frame_count: int
    duration_seconds: float


def normalize_tts_text(text: str) -> str:
    """Remove visual quote punctuation that can destabilize Raven stopping."""
    return (
        text.replace('"', "")
        .replace("\u201c", "")
        .replace("\u201d", "")
        .replace("\u201e", "")
        .replace("\u201f", "")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def plausible_audio_seconds(phrase: str) -> float:
    """Return a conservative duration ceiling for one synthesized phrase."""
    return len(phrase) / 10.0 + 4.0


def canonical_synthesis_fingerprint(inputs: Mapping[str, Any]) -> str:
    """Hash synthesis inputs using a stable, strict JSON representation."""
    try:
        payload = json.dumps(
            inputs,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fingerprint inputs are not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wav(path: Path) -> WavInfo:
    """Inspect PCM16 or Raven IEEE-float WAV data and reject malformed samples."""
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise AudioError(
            [ValidationIssue(AudioErrorCode.INVALID_WAV, str(path), str(exc))]
        ) from exc

    def invalid(message: str) -> AudioError:
        return AudioError(
            [ValidationIssue(AudioErrorCode.INVALID_WAV, str(path), message)]
        )

    if len(content) < 12 or content[:4] != b"RIFF" or content[8:12] != b"WAVE":
        raise invalid("not a RIFF/WAVE file")
    declared_size = struct.unpack_from("<I", content, 4)[0] + 8
    if declared_size > len(content):
        raise invalid("truncated RIFF payload")

    format_code = channels = sample_rate = block_align = bits = None
    data = None
    offset = 12
    while offset + 8 <= len(content):
        chunk_id = content[offset:offset + 4]
        chunk_size = struct.unpack_from("<I", content, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(content):
            raise invalid(f"truncated {chunk_id!r} chunk")
        if chunk_id == b"fmt " and chunk_size >= 16:
            format_code, channels, sample_rate, _, block_align, bits = struct.unpack_from(
                "<HHIIHH", content, chunk_start
            )
        elif chunk_id == b"data":
            data = content[chunk_start:chunk_end]
        offset = chunk_end + (chunk_size % 2)

    if None in {format_code, channels, sample_rate, block_align, bits} or data is None:
        raise invalid("missing fmt or data chunk")
    if channels <= 0 or sample_rate <= 0 or block_align <= 0:
        raise invalid("invalid channel, sample-rate, or block-alignment metadata")
    if len(data) == 0 or len(data) % block_align:
        raise invalid("empty or misaligned sample data")

    if format_code == 1 and bits == 16:
        sample_format = "pcm_s16le"
    elif format_code == 3 and bits == 32:
        sample_format = "float32le"
        if len(data) % 4:
            raise invalid("misaligned float32 sample data")
        samples = struct.iter_unpack("<f", data)
        if any(not math.isfinite(sample[0]) for sample in samples):
            raise invalid("contains non-finite float samples")
    else:
        raise invalid(f"unsupported WAV format code {format_code} with {bits} bits")

    frame_count = len(data) // block_align
    return WavInfo(
        sample_format=sample_format,
        sample_rate_hz=sample_rate,
        channels=channels,
        bits_per_sample=bits,
        frame_count=frame_count,
        duration_seconds=frame_count / sample_rate,
    )


def normalize_wav_to_pcm16(source: Path, destination: Path) -> WavInfo:
    """Convert Raven mono float32/PCM16 WAV output to canonical mono PCM16."""
    info = inspect_wav(source)
    if info.channels != 1 or info.sample_rate_hz != 24_000:
        raise AudioError(
            [ValidationIssue(
                AudioErrorCode.INVALID_WAV,
                str(source),
                "Raven output must be mono at 24000 Hz",
            )]
        )

    content = source.read_bytes()
    data = b""
    offset = 12
    while offset + 8 <= len(content):
        chunk_size = struct.unpack_from("<I", content, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if content[offset:offset + 4] == b"data":
            data = content[chunk_start:chunk_end]
            break
        offset = chunk_end + (chunk_size % 2)

    if info.sample_format == "float32le":
        pcm = bytearray()
        for (sample,) in struct.iter_unpack("<f", data):
            value = max(-1.0, min(1.0, sample))
            pcm.extend(struct.pack("<h", round(value * 32_767)))
        data = bytes(pcm)

    fmt = struct.pack("<HHIIHH", 1, 1, 24_000, 48_000, 2, 16)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    output = b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with temporary.open("wb") as stream:
            stream.write(output)
            stream.flush()
            os.fsync(stream.fileno())
        normalized = inspect_wav(temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return normalized


def _read_pcm16(path: Path) -> tuple[int, int, list[int]]:
    info = inspect_wav(path)
    if info.sample_format != "pcm_s16le":
        raise AudioError(
            [ValidationIssue(
                AudioErrorCode.INVALID_WAV,
                str(path),
                "audio processing requires PCM16 WAV audio",
            )]
        )
    content = path.read_bytes()
    offset = 12
    while offset + 8 <= len(content):
        chunk_size = struct.unpack_from("<I", content, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if content[offset:offset + 4] == b"data":
            interleaved = [sample[0] for sample in struct.iter_unpack(
                "<h", content[chunk_start:chunk_end]
            )]
            if info.channels == 1:
                samples = interleaved
            else:
                samples = [
                    round(sum(interleaved[index:index + info.channels]) / info.channels)
                    for index in range(0, len(interleaved), info.channels)
                ]
            return info.sample_rate_hz, info.channels, samples
        offset = chunk_end + (chunk_size % 2)
    raise AssertionError("inspect_wav accepted a WAV without a data chunk")


def _read_pcm16_mono(path: Path) -> tuple[int, list[int]]:
    sample_rate, channels, samples = _read_pcm16(path)
    if channels != 1:
        raise AudioError(
            [ValidationIssue(
                AudioErrorCode.INVALID_WAV,
                str(path),
                "dialogue post-processing requires mono PCM16 WAV audio",
            )]
        )
    return sample_rate, samples


def _resample_linear(
    samples: list[int], source_rate: int, destination_rate: int
) -> list[int]:
    if source_rate == destination_rate or not samples:
        return samples
    output_length = max(1, round(len(samples) * destination_rate / source_rate))
    scale = source_rate / destination_rate
    output: list[int] = []
    for output_index in range(output_length):
        position = output_index * scale
        left = min(math.floor(position), len(samples) - 1)
        right = min(left + 1, len(samples) - 1)
        fraction = position - left
        output.append(round(samples[left] * (1.0 - fraction) + samples[right] * fraction))
    return output


def _trim_and_fade_samples(samples: list[int], sample_rate: int) -> list[int]:
    if not samples or sample_rate <= 0:
        return samples
    peak = max(abs(sample) for sample in samples)
    if peak <= 3:
        return samples
    gate = peak * 0.02
    start = 0
    while start < len(samples) and abs(samples[start]) < gate:
        start += 1
    end = len(samples) - 1
    while end > start and abs(samples[end]) < gate:
        end -= 1
    if end <= start:
        return samples
    padding = round(sample_rate * 0.005)
    start = max(0, start - padding)
    end = min(len(samples) - 1, end + padding)
    trimmed = samples[start:end + 1]
    fade = min(round(sample_rate * 0.008), len(trimmed) // 2)
    for index in range(fade):
        gain = 0.5 * (1.0 - math.cos(math.pi * index / fade))
        trimmed[index] = round(trimmed[index] * gain)
        reverse_index = len(trimmed) - 1 - index
        trimmed[reverse_index] = round(trimmed[reverse_index] * gain)
    return trimmed


def _write_pcm16_atomic(path: Path, samples: list[int], sample_rate: int) -> None:
    data = struct.pack(f"<{len(samples)}h", *samples)
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    content = b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        inspect_wav(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def postprocess_dialogue_wav(source: Path, destination: Path) -> WavInfo:
    """Trim silence and fade the boundaries of canonical Raven output."""
    sample_rate, samples = _read_pcm16_mono(source)
    if sample_rate != 24_000:
        raise AudioError(
            [ValidationIssue(
                AudioErrorCode.INVALID_WAV,
                str(source),
                "Raven output must be mono at 24000 Hz",
            )]
        )
    _write_pcm16_atomic(
        destination, _trim_and_fade_samples(samples, sample_rate), sample_rate
    )
    return inspect_wav(destination)


def condition_reference_wav(source: Path, destination: Path) -> WavInfo:
    """Write a trimmed and conservatively normalized reference derivative."""
    sample_rate, _, samples = _read_pcm16(source)
    peak = max((abs(sample) for sample in samples), default=0)
    if peak <= 3:
        raise AudioError(
            [ValidationIssue(
                AudioErrorCode.VOICE_INVALID,
                str(source),
                "reference audio is effectively silent",
            )]
        )
    clipped = sum(abs(sample) >= 32_700 for sample in samples)
    if clipped / len(samples) > 0.01:
        raise AudioError(
            [ValidationIssue(
                AudioErrorCode.VOICE_INVALID,
                str(source),
                "reference audio is severely clipped",
            )]
        )
    gate = peak * 0.02
    start = 0
    while start < len(samples) and abs(samples[start]) < gate:
        start += 1
    end = len(samples) - 1
    while end > start and abs(samples[end]) < gate:
        end -= 1
    trimmed = samples[start:end + 1]
    magnitudes = sorted(abs(sample) for sample in trimmed)
    p99 = magnitudes[math.floor((len(magnitudes) - 1) * 0.99)]
    if p99 <= 3:
        raise AudioError(
            [ValidationIssue(
                AudioErrorCode.VOICE_INVALID,
                str(source),
                "reference speech level is too low",
            )]
        )
    gain = max(1.0, min(12.0, (0.9 * 32_767) / p99))
    conditioned = [
        max(-32_768, min(32_767, round(sample * gain))) for sample in trimmed
    ]
    conditioned = _resample_linear(conditioned, sample_rate, 24_000)
    _write_pcm16_atomic(destination, conditioned, 24_000)
    return inspect_wav(destination)
