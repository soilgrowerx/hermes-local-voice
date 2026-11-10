"""Tool handlers for hermes-local-voice."""

from __future__ import annotations

import json
from typing import Any

from .engine import (
    VoiceBackendError,
    VoiceInputError,
    get_last_transcript,
    transcribe_file,
)
from .recorder import start_recording, stop_recording

_PLUGIN_CONFIG: dict[str, Any] = {}


def set_plugin_config(cfg: dict[str, Any] | None) -> None:
    """Update the active runtime configuration for tool and hook handlers."""
    global _PLUGIN_CONFIG
    _PLUGIN_CONFIG = dict(cfg or {})


def get_plugin_config() -> dict[str, Any]:
    """Return a copy of the active runtime configuration."""
    return dict(_PLUGIN_CONFIG)


def handle_transcribe_audio(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    """Handler for the transcribe_audio tool."""
    params = dict(args or {})
    path = str(params.get("path") or kwargs.get("path") or "").strip()

    cfg = get_plugin_config()
    if params.get("language"):
        cfg["language"] = str(params["language"]).strip()
    if params.get("model_size"):
        cfg["model_size"] = str(params["model_size"]).strip()

    cleanup = bool(params.get("cleanup_on_success", False))

    try:
        result = transcribe_file(path, config=cfg, cleanup_source_on_success=cleanup)
        return json.dumps(result)
    except VoiceInputError as exc:
        err_payload: dict[str, Any] = {
            "status": "error",
            "error_code": exc.code,
            "error": str(exc),
        }
        if exc.temp_path:
            err_payload["preserved_audio_path"] = exc.temp_path
        return json.dumps(err_payload)
    except VoiceBackendError as exc:
        err_payload = {
            "status": "error",
            "error_code": exc.code,
            "error": str(exc),
        }
        if exc.temp_path:
            err_payload["preserved_audio_path"] = exc.temp_path
        return json.dumps(err_payload)


def handle_voice_record_start(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    """Handler for the voice_record_start tool."""
    result = start_recording()
    return json.dumps(result)


def handle_voice_record_stop(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    """Handler for the voice_record_stop tool."""
    params = dict(args or {})
    auto_tx = bool(params.get("auto_transcribe", False))
    cleanup = bool(params.get("cleanup_on_transcribe_success", False))
    result = stop_recording(
        auto_transcribe=auto_tx,
        cleanup_on_transcribe_success=cleanup,
        config=get_plugin_config(),
    )
    return json.dumps(result)


def handle_transcribe_last(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    """Handler for the transcribe_last tool."""
    last = get_last_transcript()
    if last is None:
        return json.dumps({
            "status": "empty",
            "error_code": "no_transcript",
            "message": "No transcript has been recorded yet in this session or in HERMES_HOME.",
        })
    return json.dumps(last)
