"""hermes-local-voice: Offline voice capture and transcription plugin for Hermes Agent."""

from __future__ import annotations

from typing import Any

from .cli import handle_slash_voice, handle_voice_cli, setup_voice_cli
from .hooks import on_pre_gateway_dispatch, on_pre_llm_call
from .provider import build_provider_class
from .schemas import (
    TRANSCRIBE_AUDIO_SCHEMA,
    TRANSCRIBE_LAST_SCHEMA,
    VOICE_RECORD_START_SCHEMA,
    VOICE_RECORD_STOP_SCHEMA,
)
from .tools import (
    handle_transcribe_audio,
    handle_transcribe_last,
    handle_voice_record_start,
    handle_voice_record_stop,
    set_plugin_config,
)

__version__ = "0.1.1"


def _load_context_config(ctx: Any) -> dict[str, Any]:
    """Read plugin configuration from PluginContext with defaults."""
    defaults: dict[str, Any] = {
        "model_size": "base.en",
        "compute_type": "int8",
        "language": "en",
        "model_dir": "",
        "max_file_mb": 100,
        "server_url": "",
        "prefer_server": False,
        # Off by default: pre_gateway_dispatch runs BEFORE sender authorization, so
        # enabling this lets any stranger on a connected platform trigger CPU-heavy
        # transcription. Prefer `stt.provider = hermes_local_voice` instead.
        "auto_transcribe_inbound": False,
    }
    cfg: dict[str, Any] = {}
    get_cfg = getattr(ctx, "get_config", None)
    for key, default_val in defaults.items():
        if callable(get_cfg):
            val = get_cfg(key, default_val)
            cfg[key] = default_val if val is None else val
        else:
            cfg[key] = default_val
    return cfg


def register(ctx: Any) -> None:
    """Register tools, message hooks, and CLI commands with Hermes Agent."""
    cfg = _load_context_config(ctx)
    set_plugin_config(cfg)

    ctx.register_tool(
        name="transcribe_audio",
        toolset="local_voice",
        schema=TRANSCRIBE_AUDIO_SCHEMA,
        handler=handle_transcribe_audio,
        description="Transcribe a local audio file offline with faster-whisper",
    )
    ctx.register_tool(
        name="voice_record_start",
        toolset="local_voice",
        schema=VOICE_RECORD_START_SCHEMA,
        handler=handle_voice_record_start,
        description="Start local 16 kHz mono microphone recording",
    )
    ctx.register_tool(
        name="voice_record_stop",
        toolset="local_voice",
        schema=VOICE_RECORD_STOP_SCHEMA,
        handler=handle_voice_record_stop,
        description="Stop active microphone recording and return path and duration",
    )
    ctx.register_tool(
        name="transcribe_last",
        toolset="local_voice",
        schema=TRANSCRIBE_LAST_SCHEMA,
        handler=handle_transcribe_last,
        description="Return the most recent transcript from hermes-local-voice",
    )

    provider_cls = build_provider_class()
    if provider_cls is not None:
        try:
            ctx.register_transcription_provider(provider_cls(cfg))
        except Exception:
            # Older cores without the transcription registry keep the tools usable.
            pass

    # Preferred inbound path: core selects this provider via `stt.provider` and only
    # transcribes messages it has already admitted. The legacy hook is opt-in.
    if bool(cfg.get("auto_transcribe_inbound", False)):
        ctx.register_hook("pre_gateway_dispatch", on_pre_gateway_dispatch)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)

    ctx.register_cli_command(
        name="voice",
        help="Offline voice transcription and model management",
        setup_fn=setup_voice_cli,
        handler_fn=handle_voice_cli,
        description="Transcribe audio files headlessly with faster-whisper or manage local models.",
    )
    ctx.register_command(
        name="local-voice",
        handler=handle_slash_voice,
        description="Transcribe a local audio file or inspect recent voice transcripts",
        args_hint="<file|last|models>",
    )
