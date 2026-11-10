"""Inbound message hooks for automatic offline audio attachment transcription."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .engine import SUPPORTED_EXTENSIONS, VoiceBackendError, VoiceInputError, transcribe_file
from .tools import get_plugin_config

_EXT_ALT = "|".join(ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS))
_AUDIO_PATH_RE = re.compile(
    rf"(?:@|file://)?(?P<path>(?:~|/|\./|\.\./)[^\s\"'<>|]+?\.(?:{_EXT_ALT}))\b",
    re.IGNORECASE,
)


def _extract_event_field(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def _is_audio_attachment(path_str: str, mime_type: str = "") -> bool:
    if mime_type and mime_type.lower().startswith("audio/"):
        return True
    ext = Path(path_str).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


def on_pre_gateway_dispatch(event: Any = None, **kwargs: Any) -> dict[str, Any] | None:
    """Automatically transcribe inbound audio attachments on gateway platforms."""
    cfg = get_plugin_config()
    if not bool(cfg.get("auto_transcribe_inbound", True)):
        return None

    target = event if event is not None else kwargs
    existing_text = str(_extract_event_field(target, "text", "") or "")
    if "[Voice message transcript]:" in existing_text or "[Local Voice Transcript" in existing_text:
        return None

    media_urls = list(_extract_event_field(target, "media_urls", None) or [])
    media_types = list(_extract_event_field(target, "media_types", None) or [])

    single_audio = _extract_event_field(target, "audio_path", None)
    if single_audio and single_audio not in media_urls:
        media_urls.append(str(single_audio))

    if not media_urls:
        return None

    transcripts: list[str] = []
    for idx, media_path in enumerate(media_urls):
        mime = media_types[idx] if idx < len(media_types) else ""
        path_str = str(media_path or "").strip()
        if path_str.startswith("file://"):
            path_str = path_str[len("file://"):]
        if not path_str or not _is_audio_attachment(path_str, str(mime)):
            continue
        candidate = Path(path_str).expanduser()
        if not candidate.is_file():
            continue
        try:
            res = transcribe_file(str(candidate), config=cfg)
        except (VoiceInputError, VoiceBackendError):
            continue
        text = str(res.get("text") or "").strip()
        if text:
            model = res.get("model", "local")
            duration = res.get("duration", 0.0)
            lang = res.get("language", "en")
            transcripts.append(
                f'[Local Voice Transcript ({model}, {duration}s, lang={lang})]: "{text}"'
            )

    if not transcripts:
        return None

    joined = "\n".join(transcripts)
    if existing_text.strip():
        rewritten = f"{joined}\n\n{existing_text.strip()}"
    else:
        rewritten = joined

    return {
        "action": "rewrite",
        "text": rewritten,
    }


def extract_audio_paths_from_message(user_message: str) -> list[Path]:
    """Find existing local audio file paths referenced in a chat message."""
    if not user_message:
        return []
    found: list[Path] = []
    seen: set[Path] = set()

    stripped = user_message.strip().strip("\"'")
    if stripped and Path(stripped).suffix.lower() in SUPPORTED_EXTENSIONS:
        direct = Path(stripped).expanduser()
        if direct.is_file():
            resolved = direct.resolve()
            found.append(resolved)
            seen.add(resolved)

    for match in _AUDIO_PATH_RE.finditer(user_message):
        raw_path = match.group("path")
        candidate = Path(raw_path).expanduser()
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in seen:
                found.append(resolved)
                seen.add(resolved)
    return found


def on_pre_llm_call(
    session_id: str | None = None,
    user_message: str = "",
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Inject offline transcript context when a user message attaches an audio file."""
    cfg = get_plugin_config()
    if not bool(cfg.get("auto_transcribe_inbound", True)):
        return None

    candidates: list[Path] = extract_audio_paths_from_message(str(user_message or ""))
    extra_media = kwargs.get("media_paths") or kwargs.get("attachments") or []
    if isinstance(extra_media, (list, tuple)):
        for item in extra_media:
            p = Path(str(item)).expanduser()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file():
                resolved = p.resolve()
                if resolved not in candidates:
                    candidates.append(resolved)

    if not candidates:
        return None

    blocks: list[str] = []
    for audio_path in candidates:
        try:
            res = transcribe_file(str(audio_path), config=cfg)
        except (VoiceInputError, VoiceBackendError):
            continue
        text = str(res.get("text") or "").strip()
        if text:
            model = res.get("model", "local")
            duration = res.get("duration", 0.0)
            lang = res.get("language", "en")
            blocks.append(
                f"[Local Voice Transcript for {audio_path.name} "
                f"(model={model}, duration={duration}s, lang={lang})]:\n{text}"
            )

    if not blocks:
        return None

    return {
        "context": "\n\n".join(blocks),
    }
