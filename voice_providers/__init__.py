"""Voice synthesis provider contracts shared by voice-lab workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class VoiceRequest:
    text: str
    voice_id: str
    model: str
    settings: Mapping[str, Any]
    output_path: Path
    idempotency_key: str


@dataclass(frozen=True)
class VoiceResult:
    output_path: Path
    provider_metadata: Mapping[str, Any]
    cost_usd: float | None = None


class VoiceProvider(Protocol):
    def synthesize(self, request: VoiceRequest) -> VoiceResult:
        """Synthesize one request or raise a structured provider error."""
        ...