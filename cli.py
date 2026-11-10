"""Headless CLI subcommand (`hermes voice`) and in-session `/voice` command."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .engine import (
    MODEL_DISK_MB,
    VoiceBackendError,
    VoiceInputError,
    download_model,
    get_last_transcript,
    get_model_dir,
    list_models_status,
    transcribe_file,
)
from .recorder import start_recording, stop_recording
from .tools import get_plugin_config


def setup_voice_cli(parser: argparse.ArgumentParser) -> None:
    """Configure argparse flags for `hermes voice`."""
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to a local audio file (.wav, .mp3, .ogg, .opus, .m4a, .flac, .webm) to transcribe.",
    )
    parser.add_argument(
        "--model",
        dest="model_size",
        default=None,
        help="Whisper model override (tiny.en, base.en, small.en, medium.en, large-v3-turbo).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (for example 'en') or 'auto' for language detection.",
    )
    parser.add_argument(
        "--compute-type",
        dest="compute_type",
        default=None,
        help="CTranslate2 compute precision (int8, float16, float32).",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit structured JSON output instead of plain transcript text.",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="Print the most recent transcript recorded by hermes-local-voice.",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="List supported Whisper models, approximate disk cost in MB, and cache status.",
    )
    parser.add_argument(
        "--download",
        dest="download_model_name",
        metavar="MODEL",
        default=None,
        help="Explicitly download a Whisper model into HERMES_HOME after reporting disk cost.",
    )
    parser.add_argument(
        "--record-start",
        action="store_true",
        help="Start local 16 kHz mono microphone capture.",
    )
    parser.add_argument(
        "--record-stop",
        action="store_true",
        help="Stop active local microphone capture and print the saved path and duration.",
    )


def run_voice_cli(args: argparse.Namespace) -> int:
    """Execute the `hermes voice` CLI command and return an exit code."""
    cfg = get_plugin_config()
    if getattr(args, "model_size", None):
        cfg["model_size"] = args.model_size
    if getattr(args, "language", None):
        cfg["language"] = args.language
    if getattr(args, "compute_type", None):
        cfg["compute_type"] = args.compute_type

    as_json = bool(getattr(args, "as_json", False))

    if getattr(args, "models", False):
        rows = list_models_status(cfg)
        if as_json:
            print(json.dumps({"model_dir": str(get_model_dir(cfg)), "models": rows}, indent=2))
            return 0
        print(f"Model directory: {get_model_dir(cfg)}")
        print(f"{'MODEL':<18} {'DISK COST':<12} {'STATUS':<12} {'PATH'}")
        for row in rows:
            status = "cached" if row["cached"] else "not cached"
            path_str = row["path"] or ""
            print(f"{row['model']:<18} ~{row['approx_size_mb']:<4} MB     {status:<12} {path_str}")
        return 0

    download_target = getattr(args, "download_model_name", None)
    if download_target:
        size_mb = MODEL_DISK_MB.get(download_target, 145)
        target_dir = get_model_dir(cfg)
        if not as_json:
            print(
                f"Preparing to download Whisper model '{download_target}' "
                f"(approx {size_mb} MB) into {target_dir}. Press Ctrl+C to cancel.",
                file=sys.stderr,
            )
        try:
            res = download_model(download_target, config=cfg)
        except (VoiceInputError, VoiceBackendError) as exc:
            if as_json:
                print(json.dumps({"status": "error", "error": str(exc)}))
            else:
                print(f"Error: {exc}", file=sys.stderr)
            return 1
        if as_json:
            print(json.dumps(res, indent=2))
        else:
            print(res["message"])
        return 0

    if getattr(args, "record_start", False):
        res = start_recording()
        if as_json:
            print(json.dumps(res, indent=2))
        else:
            print(res.get("message", json.dumps(res)))
        return 0 if res.get("status") in ("recording", "already_recording") else 1

    if getattr(args, "record_stop", False):
        res = stop_recording(auto_transcribe=False, config=cfg)
        if as_json:
            print(json.dumps(res, indent=2))
        else:
            print(res.get("message", json.dumps(res)))
        return 0 if res.get("status") == "stopped" else 1

    if getattr(args, "last", False):
        last = get_last_transcript()
        if last is None:
            if as_json:
                print(json.dumps({"status": "empty", "message": "No transcript recorded yet."}))
            else:
                print("No transcript recorded yet.", file=sys.stderr)
            return 1
        if as_json:
            print(json.dumps(last, indent=2))
        else:
            print(last.get("text", ""))
        return 0

    audio_file = getattr(args, "file", None)
    if not audio_file:
        print(
            "Usage: hermes voice <file> [--json] [--model MODEL] [--language LANG] "
            "| --last | --models | --download MODEL",
            file=sys.stderr,
        )
        return 2

    try:
        result = transcribe_file(str(audio_file), config=cfg)
    except (VoiceInputError, VoiceBackendError) as exc:
        if as_json:
            payload: dict[str, Any] = {
                "status": "error",
                "error_code": exc.code,
                "error": str(exc),
            }
            if exc.temp_path:
                payload["preserved_audio_path"] = exc.temp_path
            print(json.dumps(payload, indent=2))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(result.get("text", ""))
    return 0


def handle_voice_cli(args: argparse.Namespace) -> None:
    """Entry point invoked by Hermes CLI subparser dispatch."""
    code = run_voice_cli(args)
    if code != 0:
        raise SystemExit(code)


def handle_slash_voice(raw_args: str = "", **kwargs: Any) -> str:
    """In-session `/voice` slash command handler."""
    arg = (raw_args or "").strip()
    cfg = get_plugin_config()
    if not arg or arg == "last":
        last = get_last_transcript()
        if not last:
            return "No transcript has been recorded yet. Usage: /voice <path> | /voice models | /voice last"
        return (
            f"Last transcript ({last.get('model')}, {last.get('duration')}s, "
            f"lang={last.get('language')}):\n{last.get('text')}"
        )
    if arg == "models":
        rows = list_models_status(cfg)
        lines = [f"Models in {get_model_dir(cfg)}:"]
        for row in rows:
            state = "cached" if row["cached"] else "not cached"
            lines.append(f"- {row['model']} (~{row['approx_size_mb']} MB): {state}")
        return "\n".join(lines)
    try:
        res = transcribe_file(arg, config=cfg)
        return (
            f"Transcript ({res['model']}, {res['duration']}s, lang={res['language']}):\n"
            f"{res['text']}"
        )
    except (VoiceInputError, VoiceBackendError) as exc:
        return f"Voice error ({exc.code}): {exc}"
