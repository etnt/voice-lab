"""Pocket TTS Raven installation discovery and manifest validation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from audio import (
    AUDIO_POLICY_VERSION,
    AudioErrorCode,
    AudioError,
    ValidationIssue,
    condition_reference_wav,
    inspect_wav,
    normalize_wav_to_pcm16,
)
from voice_providers import VoiceRequest, VoiceResult


RAVEN_REVISION = "abd26158ab50f954616eaf42296b09c4856489d7"
INSTALLATION_SCHEMA_VERSION = 2
DR_LIBS_REVISION = "cd99e2cca5ccb48f8d000f1cd844424dcdbbaf59"
SENTENCEPIECE_VERSION = "v0.2.1"
ONNX_RUNTIME_VERSION = "1.23.2"
MODEL_REPOSITORY = "KevinAHM/pocket-tts-onnx"
MODEL_BUNDLE = "onnx/english_2026-04"
MODEL_BASE = "kyutai/pocket-tts"
MODEL_LICENSE = "CC-BY-4.0"
MODEL_IDENTITY = "content-addressed inputs pinned by tools/prepare_models.sh"
SETUP_COMMAND = "python3 tools/setup_raven.py --accept-model-terms"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RavenInstallation:
    root: Path
    executable: Path
    models: Path
    manifest_path: Path
    raven_revision: str
    onnx_runtime_version: str
    model_manifest_sha256: str


class PocketTtsRavenProvider:
    """Context-managed Raven server with standard-library HTTP synthesis."""

    def __init__(
        self,
        *,
        installation: RavenInstallation | None = None,
        base_url: str | None = None,
        voices_dir: Path | None = None,
        settings: Mapping[str, Any] | None = None,
        log_path: Path | None = None,
        startup_timeout: float = 60.0,
        request_timeout: float = 300.0,
        shutdown_timeout: float = 8.0,
    ) -> None:
        configured_url = base_url or os.environ.get("POCKET_TTS_RAVEN_BASE_URL")
        self.base_url = configured_url.rstrip("/") if configured_url else None
        if self.base_url:
            parsed = urllib.parse.urlparse(self.base_url)
            if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise _failure(
                    AudioErrorCode.RUNTIME_MISSING,
                    "POCKET_TTS_RAVEN_BASE_URL",
                    "must use an HTTP loopback address",
                )
        self.installation = installation
        self.voices_dir = voices_dir.resolve() if voices_dir else None
        self.settings = dict(settings or {})
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.shutdown_timeout = shutdown_timeout
        self._process: subprocess.Popen[str] | None = None
        self._log_stream: Any = None
        self._staged_voices: dict[str, str] = {}

    def __enter__(self) -> PocketTtsRavenProvider:
        if self.base_url is None:
            self._start_managed_server()
        self._wait_for_health()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _start_managed_server(self) -> None:
        installation = self.installation or load_raven_installation()
        if self.voices_dir is None:
            raise _failure(
                AudioErrorCode.VOICE_INVALID,
                "voices_dir",
                "managed Raven requires a writable voice directory",
            )
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"

        command = [
            str(installation.executable),
            "--server",
            "--port",
            str(port),
            "--models-dir",
            str(installation.models),
            "--tokenizer",
            str(installation.models / "tokenizer.model"),
            "--voices-dir",
            str(self.voices_dir),
            "--low-latency",
            "--trim-leading",
            "--trim-sustain",
            "15",
        ]
        option_names = {
            "precision": "--precision",
            "temperature": "--temperature",
            "lsd_steps": "--lsd-steps",
            "threads": "--threads",
        }
        for key, option in option_names.items():
            if key in self.settings:
                command.extend((option, str(self.settings[key])))
        if self.settings.get("soften_commas") is False:
            command.append("--keep-commas")

        log_path = self.log_path or self.voices_dir / "raven-server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_stream = log_path.open("a", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                command,
                cwd=installation.root,
                stdout=self._log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            self._log_stream.close()
            self._log_stream = None
            raise _failure(
                AudioErrorCode.RUNTIME_MISSING,
                str(installation.executable),
                f"cannot start Raven: {exc}",
            ) from exc

    def _wait_for_health(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_error = "server did not respond"
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                last_error = f"server exited with code {self._process.returncode}"
                break
            try:
                with urllib.request.urlopen(
                    f"{self.base_url}/health", timeout=min(1.0, self.startup_timeout)
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    if response.status == 200 and payload.get("status") == "ok":
                        return
                    last_error = f"unexpected health response: {payload!r}"
            except (OSError, ValueError, urllib.error.URLError) as exc:
                last_error = str(exc)
            time.sleep(0.05)
        self.close()
        raise _failure(
            AudioErrorCode.STARTUP_TIMEOUT,
            self.base_url or "Raven server",
            f"Raven did not become healthy within {self.startup_timeout:g}s: {last_error}",
        )

    def synthesize(self, request: VoiceRequest) -> VoiceResult:
        if self.base_url is None:
            raise _failure(
                AudioErrorCode.RUNTIME_MISSING,
                "Raven server",
                "provider must be entered before synthesis",
            )
        payload = json.dumps(
            {
                "model": request.model,
                "input": request.text,
                "voice": self._staged_voices.get(request.voice_id, request.voice_id),
                "response_format": "wav",
            },
            separators=(",", ":"),
        ).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self.base_url}/v1/audio/speech",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw_path = request.output_path.with_name(request.output_path.name + ".raw.part")
        normalized_part = request.output_path.with_name(request.output_path.name + ".part")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        request.output_path.unlink(missing_ok=True)
        request_started = time.monotonic()
        try:
            with urllib.request.urlopen(http_request, timeout=self.request_timeout) as response:
                audio = response.read()
            raw_path.write_bytes(audio)
            raw_info = inspect_wav(raw_path)
            normalized = normalize_wav_to_pcm16(raw_path, normalized_part)
            os.replace(normalized_part, request.output_path)
        except urllib.error.HTTPError as exc:
            try:
                detail = _raven_error_detail(exc.read())
            finally:
                exc.close()
            raise _failure(
                _classify_raven_error(detail),
                request.voice_id,
                f"Raven returned HTTP {exc.code}: {detail}",
            ) from exc
        except TimeoutError as exc:
            try:
                self.close()
            except AudioError:
                pass
            raise _failure(
                AudioErrorCode.REQUEST_TIMEOUT,
                request.voice_id,
                f"Raven request exceeded the {self.request_timeout:g}s timeout",
            ) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise _failure(
                AudioErrorCode.SYNTHESIS_FAILURE,
                request.voice_id,
                f"Raven request failed: {exc}",
            ) from exc
        finally:
            raw_path.unlink(missing_ok=True)
            normalized_part.unlink(missing_ok=True)
        request_seconds = time.monotonic() - request_started
        return VoiceResult(
            output_path=request.output_path,
            provider_metadata={
                "source_sample_format": raw_info.sample_format,
                "sample_rate_hz": normalized.sample_rate_hz,
                "channels": normalized.channels,
                "sample_format": normalized.sample_format,
                "frame_count": normalized.frame_count,
                "duration_seconds": normalized.duration_seconds,
                "request_seconds": request_seconds,
                "real_time_factor": request_seconds / normalized.duration_seconds,
            },
        )

    def stage_voice(self, voice_id: str, reference_audio: Path) -> Path:
        if self.voices_dir is None:
            raise _failure(
                AudioErrorCode.VOICE_INVALID,
                voice_id,
                "voice staging requires a managed voices directory",
            )
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        digest = _sha256_file(reference_audio)
        destination = self.voices_dir / f"{voice_id}{reference_audio.suffix.lower()}"
        marker = self.voices_dir / f"{voice_id}.source.json"
        current_digest = None
        current_filename = None
        current: object = None
        try:
            current = json.loads(marker.read_text(encoding="utf-8"))
            current_digest = current.get("sha256")
            current_filename = current.get("filename")
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        current_policy = current.get("audio_policy_version") if isinstance(
            current, dict
        ) else None
        if current_digest != digest or current_policy != AUDIO_POLICY_VERSION \
                or not destination.is_file():
            if current_filename and current_filename != destination.name:
                (self.voices_dir / current_filename).unlink(missing_ok=True)
            temporary = destination.with_name(destination.name + ".part")
            if reference_audio.suffix.lower() == ".wav":
                condition_reference_wav(reference_audio, temporary)
                os.replace(temporary, destination)
            else:
                shutil.copyfile(reference_audio, temporary)
                os.replace(temporary, destination)
            for suffix in ("emb", "kv"):
                (self.voices_dir / ".cache" / f"{destination.name}.{suffix}").unlink(
                    missing_ok=True
                )
            marker_part = marker.with_name(marker.name + ".part")
            marker_part.write_text(
                json.dumps(
                    {
                        "source": str(reference_audio),
                        "filename": destination.name,
                        "sha256": digest,
                        "staged_sha256": _sha256_file(destination),
                        "audio_policy_version": AUDIO_POLICY_VERSION,
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            os.replace(marker_part, marker)
        self._staged_voices[voice_id] = destination.name
        return destination

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise _failure(
                    AudioErrorCode.SHUTDOWN_TIMEOUT,
                    "Raven server",
                    f"did not stop within {self.shutdown_timeout:g}s",
                )
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None


def default_cache_root() -> Path:
    host_system = platform.system()
    if host_system == "Darwin":
        return Path.home() / "Library" / "Caches" / "voice-lab" \
            / "pocket-tts-raven"
    if host_system == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        root = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return root / "voice-lab" / "Cache" / "pocket-tts-raven"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return root / "voice-lab" / "pocket-tts-raven"


def installation_directory(cache_root: Path | None = None) -> Path:
    return (cache_root or default_cache_root()).expanduser().resolve() / RAVEN_REVISION


def _failure(code: AudioErrorCode, location: str, message: str) -> AudioError:
    return AudioError([ValidationIssue(code, location, message)])


def _raven_error_detail(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace")
    try:
        payload = json.loads(decoded)
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(error, str):
            return error
    except (json.JSONDecodeError, AttributeError):
        pass
    return decoded


def _classify_raven_error(detail: str) -> AudioErrorCode:
    normalized = detail.casefold()
    if any(term in normalized for term in (
        "failed to load audio", "voice", "reference", ".wav", ".mp3"
    )):
        return AudioErrorCode.VOICE_INVALID
    if any(term in normalized for term in (
        "model", "tokenizer", "onnx", "bos_before_voice"
    )):
        return AudioErrorCode.MODEL_INVALID
    return AudioErrorCode.SYNTHESIS_FAILURE


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _failure(AudioErrorCode.MODEL_INVALID, field, "must be a relative path")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise _failure(AudioErrorCode.MODEL_INVALID, field, "must remain inside installation")
    return path


def _verify_artifacts(
    root: Path,
    value: object,
    *,
    skipped_roles: set[str],
) -> str:
    if not isinstance(value, list) or not value:
        raise _failure(AudioErrorCode.MODEL_INVALID, "manifest.artifacts", "must be a non-empty list")
    declared: set[str] = set()
    model_artifacts: list[tuple[str, str]] = []
    valid_roles = {"executable", "runtime-library", "model", "notice"}
    for index, entry in enumerate(value):
        field = f"manifest.artifacts[{index}]"
        if not isinstance(entry, dict):
            raise _failure(AudioErrorCode.MODEL_INVALID, field, "must be an object")
        relative = entry.get("path")
        role = entry.get("role")
        size = entry.get("size")
        digest = entry.get("sha256")
        path = _manifest_path(root, relative, f"{field}.path")
        if relative == "manifest.json" or relative in declared:
            raise _failure(AudioErrorCode.MODEL_INVALID, f"{field}.path", "must be unique and exclude manifest.json")
        if role not in valid_roles:
            raise _failure(AudioErrorCode.MODEL_INVALID, f"{field}.role", f"must be one of {sorted(valid_roles)}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise _failure(AudioErrorCode.MODEL_INVALID, f"{field}.size", "must be a non-negative integer")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise _failure(AudioErrorCode.MODEL_INVALID, f"{field}.sha256", "must be a lowercase SHA-256 digest")
        declared.add(relative)
        if role == "model":
            if not relative.startswith("models/"):
                raise _failure(AudioErrorCode.MODEL_INVALID, f"{field}.path", "model artifacts must be under models/")
            model_artifacts.append((relative.removeprefix("models/"), digest))
        if role in skipped_roles:
            continue
        error_code = AudioErrorCode.MODEL_INVALID if role == "model" else AudioErrorCode.RUNTIME_MISSING
        if not path.is_file() or path.stat().st_size != size or _sha256_file(path) != digest:
            raise _failure(
                error_code,
                str(path),
                f"Raven {role} does not match its manifest. Run: {SETUP_COMMAND}",
            )

    installed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if installed != declared:
        missing = sorted(declared - installed)
        unexpected = sorted(installed - declared)
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(missing)}")
        if unexpected:
            detail.append(f"unexpected: {', '.join(unexpected)}")
        raise _failure(
            AudioErrorCode.MODEL_INVALID,
            "manifest.artifacts",
            "installation contents differ from the closed artifact inventory (" + "; ".join(detail) + ")",
        )
    aggregate = hashlib.sha256()
    for relative, digest in sorted(model_artifacts):
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest()


def load_raven_installation(
    *,
    cache_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    expected_platform: str | None = None,
    expected_architecture: str | None = None,
) -> RavenInstallation:
    """Load and verify a prepared local installation without starting Raven."""
    environment = os.environ if environ is None else environ
    root = installation_directory(cache_root)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _failure(
            AudioErrorCode.RUNTIME_MISSING,
            str(manifest_path),
            f"Raven installation is missing or unreadable: {exc}. Run: {SETUP_COMMAND}",
        ) from exc
    if not isinstance(manifest, dict):
        raise _failure(AudioErrorCode.MODEL_INVALID, str(manifest_path), "must contain an object")

    required = {
        "schema_version",
        "raven_revision",
        "platform",
        "architecture",
        "onnx_runtime_version",
        "executable",
        "executable_sha256",
        "models",
        "model_manifest_sha256",
        "model_terms_accepted_at",
        "dependencies",
        "distribution",
        "model_source",
        "artifacts",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise _failure(
            AudioErrorCode.MODEL_INVALID,
            str(manifest_path),
            f"missing required fields: {', '.join(missing)}",
        )
    if manifest["schema_version"] != INSTALLATION_SCHEMA_VERSION:
        raise _failure(
            AudioErrorCode.MODEL_INVALID,
            "manifest.schema_version",
            f"expected {INSTALLATION_SCHEMA_VERSION}, got {manifest['schema_version']!r}",
        )
    if manifest["raven_revision"] != RAVEN_REVISION:
        raise _failure(
            AudioErrorCode.MODEL_INVALID,
            "manifest.raven_revision",
            f"expected {RAVEN_REVISION}, got {manifest['raven_revision']!r}",
        )

    host_platform = expected_platform or sys.platform
    host_architecture = expected_architecture or platform.machine()
    if manifest["platform"] != host_platform or manifest["architecture"] != host_architecture:
        raise _failure(
            AudioErrorCode.RUNTIME_MISSING,
            str(manifest_path),
            "installation platform does not match this host; "
            f"expected {host_platform}/{host_architecture}. Run: {SETUP_COMMAND}",
        )

    executable_override = environment.get("POCKET_TTS_RAVEN_BIN")
    models_override = environment.get("POCKET_TTS_RAVEN_MODELS")
    skipped_roles = set()
    if executable_override:
        skipped_roles.update(("executable", "runtime-library"))
    if models_override:
        skipped_roles.add("model")
    artifact_model_hash = _verify_artifacts(
        root, manifest["artifacts"], skipped_roles=skipped_roles
    )

    distribution = manifest["distribution"]
    if not isinstance(distribution, dict) \
            or distribution.get("policy") != "local-build-only" \
            or distribution.get("publisher_signature", object()) is not None:
        raise _failure(
            AudioErrorCode.MODEL_INVALID,
            "manifest.distribution",
            "must declare local-build-only policy and no publisher signature",
        )
    expected_dependencies = {
        "dr_libs": DR_LIBS_REVISION,
        "sentencepiece": SENTENCEPIECE_VERSION,
        "onnx_runtime": ONNX_RUNTIME_VERSION,
    }
    if manifest["dependencies"] != expected_dependencies:
        raise _failure(
            AudioErrorCode.MODEL_INVALID,
            "manifest.dependencies",
            f"expected pinned dependencies {expected_dependencies!r}",
        )
    model_source = manifest["model_source"]
    expected_model_source = {
        "repository": MODEL_REPOSITORY,
        "bundle": MODEL_BUNDLE,
        "base_model": MODEL_BASE,
        "license": MODEL_LICENSE,
        "identity": MODEL_IDENTITY,
    }
    if model_source != expected_model_source:
        raise _failure(
            AudioErrorCode.MODEL_INVALID,
            "manifest.model_source",
            f"expected pinned model source {expected_model_source!r}",
        )
    executable = (
        Path(executable_override).expanduser().resolve()
        if executable_override
        else _manifest_path(root, manifest["executable"], "manifest.executable")
    )
    models = (
        Path(models_override).expanduser().resolve()
        if models_override
        else _manifest_path(root, manifest["models"], "manifest.models")
    )

    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise _failure(
            AudioErrorCode.RUNTIME_MISSING,
            str(executable),
            f"Raven executable is missing or not executable. Run: {SETUP_COMMAND}",
        )
    if not executable_override:
        expected_hash = manifest["executable_sha256"]
        if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash) \
                or _sha256_file(executable) != expected_hash:
            raise _failure(
                AudioErrorCode.RUNTIME_MISSING,
                str(executable),
                f"Raven executable hash does not match its manifest. Run: {SETUP_COMMAND}",
            )
    if not models.is_dir():
        raise _failure(
            AudioErrorCode.MODEL_INVALID,
            str(models),
            f"Raven model directory is missing. Run: {SETUP_COMMAND}",
        )
    model_hash = manifest["model_manifest_sha256"]
    if not isinstance(model_hash, str) or not SHA256_PATTERN.fullmatch(model_hash):
        raise _failure(
            AudioErrorCode.MODEL_INVALID,
            "manifest.model_manifest_sha256",
            "must be a lowercase SHA-256 digest",
        )
    if not models_override and artifact_model_hash != model_hash:
        raise _failure(
            AudioErrorCode.MODEL_INVALID,
            str(models),
            f"Raven model files do not match their manifest. Run: {SETUP_COMMAND}",
        )

    onnx_version = manifest["onnx_runtime_version"]
    if onnx_version != ONNX_RUNTIME_VERSION:
        raise _failure(
            AudioErrorCode.MODEL_INVALID,
            "manifest.onnx_runtime_version",
            f"expected {ONNX_RUNTIME_VERSION}, got {onnx_version!r}",
        )

    return RavenInstallation(
        root=root,
        executable=executable,
        models=models,
        manifest_path=manifest_path,
        raven_revision=RAVEN_REVISION,
        onnx_runtime_version=onnx_version,
        model_manifest_sha256=model_hash,
    )