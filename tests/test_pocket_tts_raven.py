import hashlib
import http.server
import json
import struct
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest import mock

from audio import AudioErrorCode, AudioError
from voice_providers.pocket_tts_raven import (
    DR_LIBS_REVISION,
    INSTALLATION_SCHEMA_VERSION,
    MODEL_BASE,
    MODEL_BUNDLE,
    MODEL_IDENTITY,
    MODEL_LICENSE,
    MODEL_REPOSITORY,
    PocketTtsRavenProvider,
    RAVEN_REVISION,
    RavenInstallation,
    installation_directory,
    load_raven_installation,
)
from voice_providers import VoiceRequest


def float32_wav(samples: tuple[float, ...]) -> bytes:
    data = struct.pack(f"<{len(samples)}f", *samples)
    fmt = struct.pack("<HHIIHH", 3, 1, 24_000, 96_000, 4, 32)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body


def write_pcm16_wav(path: Path, amplitude: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        samples = [0] * 240 + [amplitude] * 2_400 + [0] * 240
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def model_manifest_sha256(models: Path) -> str:
    aggregate = hashlib.sha256()
    for path in sorted(entry for entry in models.rglob("*") if entry.is_file()):
        relative = path.relative_to(models).as_posix()
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest()


def artifact_entry(root: Path, relative: str, role: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "role": role,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class FakeRavenHandler(http.server.BaseHTTPRequestHandler):
    requests: list[dict[str, object]] = []
    error_payload: bytes | None = None

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        payload = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        if self.path != "/v1/audio/speech":
            self.send_error(404)
            return
        length = int(self.headers["Content-Length"])
        self.requests.append(json.loads(self.rfile.read(length)))
        if self.error_payload is not None:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self.error_payload)))
            self.end_headers()
            self.wfile.write(self.error_payload)
            return
        payload = float32_wav((-1.0, -0.25, 0.25, 1.0))
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass


class RavenInstallationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.cache_root = Path(self.temporary_directory.name)
        self.installation = installation_directory(self.cache_root)
        (self.installation / "bin").mkdir(parents=True)
        (self.installation / "models").mkdir()
        self.executable = self.installation / "bin" / "pocket-tts"
        self.executable.write_bytes(b"test executable")
        self.executable.chmod(0o755)
        (self.installation / "bin" / "runtime.dylib").write_bytes(b"runtime")
        (self.installation / "models" / "tokenizer.model").write_bytes(b"tokenizer")
        (self.installation / "models" / "bos_before_voice.npy").write_bytes(b"bos")
        (self.installation / "models" / "model.onnx").write_bytes(b"model")
        self.model_hash = model_manifest_sha256(self.installation / "models")
        self.manifest = {
            "schema_version": INSTALLATION_SCHEMA_VERSION,
            "raven_revision": RAVEN_REVISION,
            "platform": "test-platform",
            "architecture": "test-architecture",
            "onnx_runtime_version": "1.23.2",
            "executable": "bin/pocket-tts",
            "executable_sha256": hashlib.sha256(b"test executable").hexdigest(),
            "models": "models",
            "model_manifest_sha256": self.model_hash,
            "model_terms_accepted_at": "2026-09-08T12:00:00Z",
            "dependencies": {
                "dr_libs": DR_LIBS_REVISION,
                "sentencepiece": "v0.2.1",
                "onnx_runtime": "1.23.2",
            },
            "distribution": {
                "policy": "local-build-only",
                "publisher_signature": None,
            },
            "model_source": {
                "repository": MODEL_REPOSITORY,
                "bundle": MODEL_BUNDLE,
                "base_model": MODEL_BASE,
                "license": MODEL_LICENSE,
                "identity": MODEL_IDENTITY,
            },
            "artifacts": [
                artifact_entry(self.installation, "bin/pocket-tts", "executable"),
                artifact_entry(self.installation, "bin/runtime.dylib", "runtime-library"),
                artifact_entry(self.installation, "models/bos_before_voice.npy", "model"),
                artifact_entry(self.installation, "models/model.onnx", "model"),
                artifact_entry(self.installation, "models/tokenizer.model", "model"),
            ],
        }
        (self.installation / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loads_valid_versioned_installation(self) -> None:
        result = load_raven_installation(
            cache_root=self.cache_root,
            environ={},
            expected_platform="test-platform",
            expected_architecture="test-architecture",
        )

        self.assertEqual(result.executable, self.executable)
        self.assertEqual(result.models, self.installation / "models")
        self.assertEqual(result.raven_revision, RAVEN_REVISION)
        self.assertEqual(result.model_manifest_sha256, self.model_hash)

    def test_rejects_executable_hash_mismatch(self) -> None:
        self.executable.write_bytes(b"changed")

        with self.assertRaises(AudioError) as raised:
            load_raven_installation(
                cache_root=self.cache_root,
                environ={},
                expected_platform="test-platform",
                expected_architecture="test-architecture",
            )

        self.assertEqual(
            raised.exception.issues[0].code, AudioErrorCode.RUNTIME_MISSING
        )

    def test_rejects_model_hash_mismatch(self) -> None:
        (self.installation / "models" / "model.onnx").write_bytes(b"changed")

        with self.assertRaises(AudioError) as raised:
            load_raven_installation(
                cache_root=self.cache_root,
                environ={},
                expected_platform="test-platform",
                expected_architecture="test-architecture",
            )

        self.assertEqual(raised.exception.issues[0].code, AudioErrorCode.MODEL_INVALID)

    def test_rejects_runtime_library_hash_mismatch(self) -> None:
        (self.installation / "bin" / "runtime.dylib").write_bytes(b"changed")

        with self.assertRaises(AudioError) as raised:
            load_raven_installation(
                cache_root=self.cache_root,
                environ={},
                expected_platform="test-platform",
                expected_architecture="test-architecture",
            )

        self.assertEqual(raised.exception.issues[0].code, AudioErrorCode.RUNTIME_MISSING)

    def test_rejects_artifact_path_outside_installation(self) -> None:
        self.manifest["artifacts"][0]["path"] = "../pocket-tts"
        (self.installation / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

        with self.assertRaises(AudioError) as raised:
            load_raven_installation(
                cache_root=self.cache_root,
                environ={},
                expected_platform="test-platform",
                expected_architecture="test-architecture",
            )

        self.assertEqual(raised.exception.issues[0].code, AudioErrorCode.MODEL_INVALID)

    def test_rejects_unpinned_model_source(self) -> None:
        self.manifest["model_source"]["repository"] = "someone/other-model"
        (self.installation / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

        with self.assertRaises(AudioError) as raised:
            load_raven_installation(
                cache_root=self.cache_root,
                environ={},
                expected_platform="test-platform",
                expected_architecture="test-architecture",
            )

        self.assertEqual(raised.exception.issues[0].code, AudioErrorCode.MODEL_INVALID)

    def test_reports_missing_installation_with_setup_command(self) -> None:
        (self.installation / "manifest.json").unlink()

        with self.assertRaisesRegex(
            AudioError, r"tools/setup_raven\.py --accept-model-terms"
        ) as raised:
            load_raven_installation(cache_root=self.cache_root, environ={})

        self.assertEqual(
            raised.exception.issues[0].code, AudioErrorCode.RUNTIME_MISSING
        )

    def test_explicit_paths_override_manifest_artifact_locations(self) -> None:
        alternate_executable = self.cache_root / "custom-pocket-tts"
        alternate_models = self.cache_root / "custom-models"
        alternate_executable.write_bytes(b"custom")
        alternate_executable.chmod(0o755)
        alternate_models.mkdir()

        result = load_raven_installation(
            cache_root=self.cache_root,
            environ={
                "POCKET_TTS_RAVEN_BIN": str(alternate_executable),
                "POCKET_TTS_RAVEN_MODELS": str(alternate_models),
            },
            expected_platform="test-platform",
            expected_architecture="test-architecture",
        )

        self.assertEqual(result.executable, alternate_executable.resolve())
        self.assertEqual(result.models, alternate_models.resolve())


class RavenProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeRavenHandler.requests = []
        FakeRavenHandler.error_payload = None
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeRavenHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary_directory.cleanup()

    def test_external_server_synthesis_publishes_pcm16_atomically(self) -> None:
        output = Path(self.temporary_directory.name) / "dialogue.wav"
        provider = PocketTtsRavenProvider(
            base_url=f"http://127.0.0.1:{self.server.server_port}",
        )
        request = VoiceRequest(
            text="The signal is clean.",
            voice_id="narrator-v1",
            model="pocket-tts-raven",
            settings={},
            output_path=output,
            idempotency_key="test-key",
        )

        with provider:
            result = provider.synthesize(request)

        self.assertEqual(result.output_path, output)
        self.assertTrue(output.is_file())
        self.assertFalse(output.with_suffix(".wav.part").exists())
        self.assertEqual(output.read_bytes()[20:22], b"\x01\x00")
        self.assertEqual(
            FakeRavenHandler.requests,
            [{
                "model": "pocket-tts-raven",
                "input": "The signal is clean.",
                "voice": "narrator-v1",
                "response_format": "wav",
            }],
        )

    def test_rejects_non_loopback_server(self) -> None:
        with self.assertRaisesRegex(AudioError, "loopback"):
            PocketTtsRavenProvider(base_url="https://example.com")

    def test_classifies_missing_reference_as_voice_invalid(self) -> None:
        FakeRavenHandler.error_payload = (
            b'{"error":{"message":"Failed to load audio: narrator.wav",'
            b'"type":"server_error"}}'
        )
        output = Path(self.temporary_directory.name) / "dialogue.wav"
        provider = PocketTtsRavenProvider(
            base_url=f"http://127.0.0.1:{self.server.server_port}",
        )
        request = VoiceRequest(
            text="Test.",
            voice_id="narrator-v1",
            model="pocket-tts-raven",
            settings={},
            output_path=output,
            idempotency_key="test-key",
        )

        with self.assertRaises(AudioError) as raised:
            provider.synthesize(request)

        self.assertEqual(raised.exception.issues[0].code, AudioErrorCode.VOICE_INVALID)
        self.assertFalse(output.exists())

    def test_classifies_request_timeout_and_removes_partial_files(self) -> None:
        output = Path(self.temporary_directory.name) / "dialogue.wav"
        provider = PocketTtsRavenProvider(base_url="http://127.0.0.1:1")
        request = VoiceRequest(
            text="Test.",
            voice_id="narrator-v1",
            model="pocket-tts-raven",
            settings={},
            output_path=output,
            idempotency_key="test-key",
        )

        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(AudioError) as raised:
                provider.synthesize(request)

        self.assertEqual(
            raised.exception.issues[0].code, AudioErrorCode.REQUEST_TIMEOUT
        )
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name(output.name + ".raw.part").exists())
        self.assertFalse(output.with_name(output.name + ".part").exists())

    def test_staging_changed_reference_invalidates_native_cache(self) -> None:
        root = Path(self.temporary_directory.name)
        reference = root / "reference.wav"
        voices = root / "staged"
        write_pcm16_wav(reference, 1_000)
        provider = PocketTtsRavenProvider(
            base_url=f"http://127.0.0.1:{self.server.server_port}",
            voices_dir=voices,
        )

        provider.stage_voice("narrator", reference)
        cache = voices / ".cache"
        cache.mkdir()
        embedding = cache / "narrator.wav.emb"
        key_values = cache / "narrator.wav.kv"
        embedding.write_bytes(b"embedding")
        key_values.write_bytes(b"key-values")
        provider.stage_voice("narrator", reference)
        self.assertTrue(embedding.exists())
        self.assertTrue(key_values.exists())

        write_pcm16_wav(reference, 2_000)
        provider.stage_voice("narrator", reference)
        self.assertFalse(embedding.exists())
        self.assertFalse(key_values.exists())

    def test_managed_command_preserves_commas_when_requested(self) -> None:
        root = Path(self.temporary_directory.name)
        executable = root / "pocket-tts"
        models = root / "models"
        executable.write_bytes(b"binary")
        models.mkdir()
        installation = RavenInstallation(
            root=root,
            executable=executable,
            models=models,
            manifest_path=root / "manifest.json",
            raven_revision=RAVEN_REVISION,
            onnx_runtime_version="1.23.2",
            model_manifest_sha256="a" * 64,
        )
        process = mock.Mock()
        process.poll.return_value = 0
        provider = PocketTtsRavenProvider(
            installation=installation,
            voices_dir=root / "voices",
            settings={"temperature": 0.2, "soften_commas": False},
        )

        with mock.patch("subprocess.Popen", return_value=process) as popen:
            provider._start_managed_server()

        command = popen.call_args.args[0]
        self.assertIn("--keep-commas", command)
        self.assertEqual(command[command.index("--temperature") + 1], "0.2")
        provider.close()


if __name__ == "__main__":
    unittest.main()