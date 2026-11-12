"""Speech-to-text provider so Hermes core transcribes admitted messages.

Core review point 1: the previous inbound path ran in ``pre_gateway_dispatch``,
which fires before sender authorization, so any stranger on a connected platform
could make the host run CPU-heavy transcription. Registering this provider means
the user selects it with ``stt.provider`` and core calls it only for messages it
has already admitted.
"""

from __future__ import annotations

from typing import Any, Optional

from .engine import VoiceBackendError, VoiceInputError, transcribe_file

PROVIDER_NAME = "hermes_local_voice"


def _envelope(success: bool, transcript: str = "", error: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": success,
        "transcript": transcript,
        "provider": PROVIDER_NAME,
    }
    if not success:
        payload["error"] = error or "transcription failed"
    return payload


def build_provider_class():
    """Return the concrete provider class, or None when core has no such ABC."""
    try:
        from agent.transcription_provider import TranscriptionProvider  # type: ignore
    except Exception:
        return None

    class HermesLocalVoiceProvider(TranscriptionProvider):
        """Offline faster-whisper transcription, no network, no cloud."""

        def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
            self._config = dict(config or {})

        @property
        def name(self) -> str:
            return PROVIDER_NAME

        def is_available(self) -> bool:
            return True

        def list_models(self) -> list[dict[str, Any]]:
            from .engine import MODEL_DISK_MB

            return [{"id": key, "display": key} for key in MODEL_DISK_MB]

        def default_model(self) -> Optional[str]:
            return str(self._config.get("model_size") or "base.en")

        def transcribe(
            self,
            file_path: str,
            *,
            model: Optional[str] = None,
            language: Optional[str] = None,
            **extra: Any,
        ) -> dict[str, Any]:
            cfg = dict(self._config)
            if model:
                cfg["model_size"] = str(model)
            if language:
                cfg["language"] = str(language)
            # A provider must never transcribe a path outside what core handed it,
            # and it must never delete anything: cleanup stays opt-in and bounded.
            try:
                result = transcribe_file(str(file_path), config=cfg, cleanup_source_on_success=False)
            except (VoiceInputError, VoiceBackendError) as exc:
                return _envelope(False, error=str(exc))
            except Exception as exc:  # must not raise into core
                return _envelope(False, error=f"{type(exc).__name__}: {exc}")

            text = str(result.get("text") or "").strip()
            if not text:
                return _envelope(False, error="empty transcript")
            return _envelope(True, transcript=text)

    return HermesLocalVoiceProvider
