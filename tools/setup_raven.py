#!/usr/bin/env python3
"""Validate prerequisites for the pinned Pocket TTS Raven installation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from audio import AudioError  # noqa: E402
from voice_providers.pocket_tts_raven import (  # noqa: E402
    DR_LIBS_REVISION,
    INSTALLATION_SCHEMA_VERSION,
    MODEL_BASE,
    MODEL_BUNDLE,
    MODEL_IDENTITY,
    MODEL_LICENSE,
    MODEL_REPOSITORY,
    ONNX_RUNTIME_VERSION,
    PocketTtsRavenProvider,
    RAVEN_REVISION,
    RavenInstallation,
    SENTENCEPIECE_VERSION,
    default_cache_root,
    installation_directory,
    load_raven_installation,
)


DR_LIBS_REPOSITORY = "https://github.com/mackron/dr_libs.git"


class SetupPreflightError(Exception):
    pass


def require_command(name: str) -> str:
    command = shutil.which(name)
    if command is None:
        raise SetupPreflightError(f"required command not found: {name}")
    return command


def git_output(source: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            [require_command("git"), "-C", str(source), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise SetupPreflightError(f"cannot inspect Raven source: {detail}") from exc
    return result.stdout.strip()


def validate_source_checkout(source: Path) -> None:
    if not source.is_dir():
        raise SetupPreflightError(
            f"Raven submodule is missing: {source}\n"
            "Run: git submodule update --init --recursive"
        )
    revision = git_output(source, "rev-parse", "HEAD")
    if revision != RAVEN_REVISION:
        raise SetupPreflightError(
            f"Raven revision mismatch: expected {RAVEN_REVISION}, got {revision}"
        )
    changes = git_output(source, "status", "--short", "--untracked-files=no")
    if changes:
        raise SetupPreflightError(
            "Raven source has tracked modifications; setup requires the pinned "
            "clean submodule"
        )
    for required in ("CMakeLists.txt", "LICENSE", "THIRD_PARTY_NOTICES.md",
                     "tools/prepare_models.sh"):
        if not (source / required).is_file():
            raise SetupPreflightError(f"Raven source is incomplete: missing {required}")


SUPPORTED_HOSTS = {
    ("darwin", "arm64"): "macOS arm64",
    ("linux", "x86_64"): "Linux x86_64",
}


def validate_host() -> None:
    host = (sys.platform, platform.machine())
    if host not in SUPPORTED_HOSTS:
        supported = ", ".join(sorted(SUPPORTED_HOSTS.values()))
        raise SetupPreflightError(
            f"unsupported host {sys.platform}/{platform.machine()}; "
            f"supported platforms: {supported}"
        )
    for command in ("cmake", "git", "uv"):
        require_command(command)
    if sys.platform == "darwin":
        try:
            subprocess.run(
                ["xcrun", "--find", "clang++"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise SetupPreflightError(
                "Apple clang++ was not found; install the Xcode command line tools"
            ) from exc
    else:
        require_command("g++")


def run_logged(command: list[str], *, cwd: Path, log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("+ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise SetupPreflightError(
            f"command failed with exit code {result.returncode}; see {log_path}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest_sha256(models: Path) -> str:
    required = (models / "tokenizer.model", models / "bos_before_voice.npy")
    missing = [path.name for path in required if not path.is_file()]
    onnx_files = sorted(models.glob("*.onnx"))
    if missing or not onnx_files:
        detail = ", ".join(missing) if missing else "ONNX model files"
        raise SetupPreflightError(f"prepared Raven models are incomplete: {detail}")
    aggregate = hashlib.sha256()
    for path in sorted(entry for entry in models.rglob("*") if entry.is_file()):
        relative = path.relative_to(models).as_posix()
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(sha256_file(path).encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest()


def artifact_inventory(root: Path) -> list[dict[str, object]]:
    artifacts = []
    for path in sorted(entry for entry in root.rglob("*") if entry.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        if relative == "bin/pocket-tts":
            role = "executable"
        elif relative.startswith("bin/"):
            role = "runtime-library"
        elif relative.startswith("models/"):
            role = "model"
        elif relative.startswith("licenses/"):
            role = "notice"
        else:
            raise SetupPreflightError(f"unclassified installation artifact: {relative}")
        artifacts.append({
            "path": relative,
            "role": role,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return artifacts


def copy_source(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", "models", "build", "*.tmp", "*.bak"),
    )


def prepare_dr_libs(workspace: Path, log_path: Path) -> Path:
    checkout = workspace / "dependencies" / "dr_libs"
    checkout.mkdir(parents=True)
    run_logged(["git", "init", "--quiet"], cwd=checkout, log_path=log_path)
    run_logged(
        ["git", "remote", "add", "origin", DR_LIBS_REPOSITORY],
        cwd=checkout,
        log_path=log_path,
    )
    run_logged(
        ["git", "fetch", "--depth=1", "origin", DR_LIBS_REVISION],
        cwd=checkout,
        log_path=log_path,
    )
    run_logged(
        ["git", "checkout", "--detach", "--force", "FETCH_HEAD"],
        cwd=checkout,
        log_path=log_path,
    )
    revision = git_output(checkout, "rev-parse", "HEAD")
    if revision != DR_LIBS_REVISION:
        raise SetupPreflightError(
            f"dr_libs revision mismatch: expected {DR_LIBS_REVISION}, got {revision}"
        )
    return checkout


def publish_installation(source: Path, cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    target = installation_directory(cache_root)
    log_path = cache_root / f"setup-{RAVEN_REVISION}.log"
    log_path.write_text("", encoding="utf-8")
    workspace = Path(tempfile.mkdtemp(prefix=f".build-{RAVEN_REVISION[:12]}-", dir=cache_root))
    staging = Path(tempfile.mkdtemp(prefix=f".install-{RAVEN_REVISION[:12]}-", dir=cache_root))
    backup = target.with_name(target.name + ".previous")
    try:
        source_copy = workspace / "source"
        copy_source(source, source_copy)
        dr_libs = prepare_dr_libs(workspace, log_path)
        build = workspace / "build"
        binary_dir = staging / "bin"
        binary_dir.mkdir(parents=True)
        configure = [
            require_command("cmake"),
            "-S", str(source_copy),
            "-B", str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPTT_OUTPUT_DIR={binary_dir}",
            f"-DFETCHCONTENT_SOURCE_DIR_DR_LIBS={dr_libs}",
        ]
        if sys.platform == "linux":
            configure.extend([
                f"-DCMAKE_CXX_COMPILER={require_command('g++')}",
                # The staged binary is moved into bin/ next to the ONNX
                # Runtime shared library that the build copies there, so it
                # must resolve that library via $ORIGIN.
                "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            ])
        run_logged(configure, cwd=workspace, log_path=log_path)
        run_logged(
            [require_command("cmake"), "--build", str(build), "--parallel"],
            cwd=workspace,
            log_path=log_path,
        )
        executable = binary_dir / "pocket-tts"
        if not executable.is_file():
            raise SetupPreflightError(
                f"build completed without expected executable: {executable}"
            )
        if sys.platform == "darwin":
            run_logged(
                [require_command("install_name_tool"), "-add_rpath", "@executable_path", str(executable)],
                cwd=workspace,
                log_path=log_path,
            )
        run_logged(
            [str(source_copy / "tools" / "prepare_models.sh")],
            cwd=source_copy,
            log_path=log_path,
        )
        shutil.move(str(source_copy / "models"), str(staging / "models"))
        licenses = staging / "licenses"
        licenses.mkdir()
        shutil.copy2(source_copy / "LICENSE", licenses / "pocket-tts-raven-LICENSE.txt")
        shutil.copy2(
            source_copy / "THIRD_PARTY_NOTICES.md",
            licenses / "pocket-tts-raven-THIRD_PARTY_NOTICES.md",
        )
        shutil.copy2(
            REPOSITORY_ROOT / "docs" / "pocket-tts-raven-distribution-review.md",
            licenses / "voice-lab-distribution-review.md",
        )
        model_hash = model_manifest_sha256(staging / "models")
        manifest = {
            "schema_version": INSTALLATION_SCHEMA_VERSION,
            "raven_revision": RAVEN_REVISION,
            "platform": sys.platform,
            "architecture": platform.machine(),
            "onnx_runtime_version": ONNX_RUNTIME_VERSION,
            "executable": "bin/pocket-tts",
            "executable_sha256": sha256_file(executable),
            "models": "models",
            "model_manifest_sha256": model_hash,
            "model_terms_accepted_at": datetime.now(timezone.utc).isoformat(),
            "dependencies": {
                "dr_libs": DR_LIBS_REVISION,
                "sentencepiece": SENTENCEPIECE_VERSION,
                "onnx_runtime": ONNX_RUNTIME_VERSION,
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
            "artifacts": artifact_inventory(staging),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        staged_installation = RavenInstallation(
            root=staging,
            executable=executable,
            models=staging / "models",
            manifest_path=staging / "manifest.json",
            raven_revision=RAVEN_REVISION,
            onnx_runtime_version=ONNX_RUNTIME_VERSION,
            model_manifest_sha256=model_hash,
        )
        with tempfile.TemporaryDirectory(dir=cache_root) as voices:
            with PocketTtsRavenProvider(
                installation=staged_installation,
                voices_dir=Path(voices),
                startup_timeout=60,
            ):
                pass

        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except BaseException:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)

        load_raven_installation(cache_root=cache_root)
        return target
    except BaseException:
        print(f"Setup log preserved at: {log_path}", file=sys.stderr)
        raise
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned Pocket TTS Raven local runtime."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate source and host prerequisites without downloads or builds",
    )
    parser.add_argument(
        "--check-installation",
        action="store_true",
        help="validate an existing installation manifest and artifacts",
    )
    parser.add_argument(
        "--accept-model-terms",
        action="store_true",
        help="explicitly accept the upstream model license and prohibited-use terms",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=REPOSITORY_ROOT / "vendor" / "pocket-tts-raven",
        help="pinned Raven source checkout",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=default_cache_root(),
        help="parent directory for versioned Raven installations",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_installation:
        installation = load_raven_installation(cache_root=args.cache_root)
        print(f"Raven installation valid: {installation.root}")
        return 0

    validate_host()
    validate_source_checkout(args.source.resolve())
    print(f"Raven source valid: {RAVEN_REVISION}")
    print(f"Installation target: {args.cache_root.expanduser().resolve()}")

    if args.validate_only:
        try:
            installation = load_raven_installation(cache_root=args.cache_root)
        except AudioError:
            print("Installed runtime: not present")
        else:
            print(f"Installed runtime valid: {installation.root}")
        print("Offline preflight passed; no files were downloaded or built.")
        return 0

    if not args.accept_model_terms:
        raise SetupPreflightError(
            "model download requires explicit acceptance: "
            "--accept-model-terms"
        )
    installation = publish_installation(args.source.resolve(), args.cache_root.resolve())
    print(f"Raven installation ready: {installation}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SetupPreflightError, AudioError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc