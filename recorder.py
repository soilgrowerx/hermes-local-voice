"""Local microphone capture using PipeWire (pw-record), ALSA (arecord), or ffmpeg."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
import wave
from pathlib import Path
from typing import Any

from .runtime import minimal_env
from .engine import (
    VoiceBackendError,
    VoiceInputError,
    find_binary,
    get_plugin_data_dir,
    transcribe_file,
)


def _state_file() -> Path:
    return get_plugin_data_dir() / "recording_state.json"


def _recordings_dir() -> Path:
    rec_dir = get_plugin_data_dir() / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    return rec_dir


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _read_recording_state() -> dict[str, Any] | None:
    sf = _state_file()
    if not sf.is_file():
        return None
    try:
        return json.loads(sf.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_recording_state(state: dict[str, Any] | None) -> None:
    sf = _state_file()
    if state is None:
        sf.unlink(missing_ok=True)
        return
    tmp = sf.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, sf)


def detect_capture_command(output_path: Path) -> tuple[list[str], str] | None:
    """Build a 16 kHz mono WAV capture command for the current platform."""
    if os.environ.get("HERMES_VOICE_DISABLE_MIC", "").strip() in ("1", "true", "yes"):
        return None

    pw_record = find_binary("pw-record")
    if pw_record:
        return (
            [
                pw_record,
                "--rate=16000",
                "--channels=1",
                "--format=s16",
                str(output_path),
            ],
            "pw-record",
        )

    arecord = find_binary("arecord")
    if arecord:
        return (
            [
                arecord,
                "-q",
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-c",
                "1",
                str(output_path),
            ],
            "arecord",
        )

    ffmpeg = find_binary("ffmpeg")
    if ffmpeg:
        if sys.platform == "darwin":
            return (
                [
                    ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "avfoundation",
                    "-i",
                    ":0",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(output_path),
                ],
                "ffmpeg-avfoundation",
            )
        return (
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-f",
                "pulse",
                "-i",
                "default",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            "ffmpeg-pulse",
        )

    return None


def start_recording() -> dict[str, Any]:
    """Start local microphone capture and return structured session details.

    Degrades cleanly with status='unavailable' when no capture binary or microphone
    audio session is available on the host.
    """
    existing = _read_recording_state()
    if existing is not None:
        pid = int(existing.get("pid") or 0)
        if _pid_is_alive(pid):
            started = float(existing.get("started_at") or time.time())
            return {
                "status": "already_recording",
                "pid": pid,
                "path": existing.get("path"),
                "backend": existing.get("backend"),
                "elapsed_seconds": round(max(0.0, time.time() - started), 2),
                "message": f"Voice recording is already active (PID {pid}). Call voice_record_stop to finish.",
            }
        _write_recording_state(None)

    out_path = _recordings_dir() / f"voice_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
    detected = detect_capture_command(out_path)
    if detected is None:
        return {
            "status": "unavailable",
            "error_code": "microphone_unavailable",
            "path": None,
            "duration": 0.0,
            "message": (
                "Microphone capture is unavailable on this host: none of pw-record, arecord, "
                "or ffmpeg audio capture could be initialized. Pass an existing audio file to "
                "transcribe_audio(path) instead."
            ),
        }

    cmd, backend_name = detected
    err_log = out_path.with_suffix(".err")
    try:
        with err_log.open("wb") as err_fh:
            proc = subprocess.Popen(
                  cmd,
                  stdin=subprocess.DEVNULL,
                  stdout=subprocess.DEVNULL,
                  stderr=err_fh,
                  start_new_session=True,
                  env=minimal_env("capture"),
              )
    except OSError as exc:
        err_log.unlink(missing_ok=True)
        return {
            "status": "unavailable",
            "error_code": "microphone_unavailable",
            "path": None,
            "duration": 0.0,
            "message": f"Failed to start {backend_name} microphone capture: {exc}",
        }

    time.sleep(0.18)
    exit_code = proc.poll()
    if exit_code is not None:
        stderr_text = ""
        if err_log.is_file():
            stderr_text = err_log.read_text(encoding="utf-8", errors="replace").strip()
            err_log.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
        detail = f": {stderr_text}" if stderr_text else f" (exit code {exit_code})"
        return {
            "status": "unavailable",
            "error_code": "microphone_unavailable",
            "path": None,
            "duration": 0.0,
            "message": (
                f"Microphone capture via {backend_name} exited immediately{detail}. "
                "Ensure a microphone and PipeWire or PulseAudio session are active, or use transcribe_audio(path)."
            ),
        }

    state = {
        "pid": proc.pid,
        "path": str(out_path),
        "err_log": str(err_log),
        "backend": backend_name,
        "started_at": time.time(),
    }
    _write_recording_state(state)
    return {
        "status": "recording",
        "pid": proc.pid,
        "path": str(out_path),
        "backend": backend_name,
        "message": f"Started 16 kHz mono voice capture via {backend_name} (PID {proc.pid}).",
    }


def _compute_wav_duration_safe(path: Path, fallback_elapsed: float) -> float:
    if not path.is_file() or path.stat().st_size < 44:
        return round(max(0.0, fallback_elapsed), 3)
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            frames = wf.getnframes()
            if rate > 0 and frames > 0:
                return round(frames / float(rate), 3)
    except wave.Error:
        pass
    return round(max(0.0, fallback_elapsed), 3)


def stop_recording(
    *,
    auto_transcribe: bool = False,
    cleanup_on_transcribe_success: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stop the active microphone recording and return the audio path and duration."""
    state = _read_recording_state()
    if state is None:
        return {
            "status": "error",
            "error_code": "not_recording",
            "path": None,
            "duration": 0.0,
            "message": "No active voice recording session found. Call voice_record_start first.",
        }

    pid = int(state.get("pid") or 0)
    audio_path = Path(str(state.get("path") or ""))
    err_log = Path(str(state.get("err_log") or "")) if state.get("err_log") else None
    started_at = float(state.get("started_at") or time.time())
    backend = str(state.get("backend") or "pw-record")

    if _pid_is_alive(pid):
        try:
            os.kill(pid, signal.SIGINT)
        except OSError:
            pass
        deadline = time.time() + 3.0
        while time.time() < deadline and _pid_is_alive(pid):
            time.sleep(0.05)
        if _pid_is_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
            time.sleep(0.1)

    _write_recording_state(None)
    if err_log and err_log.exists():
        err_log.unlink(missing_ok=True)

    elapsed = max(0.0, time.time() - started_at)
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        return {
            "status": "error",
            "error_code": "empty_recording",
            "path": str(audio_path) if audio_path else None,
            "duration": 0.0,
            "message": "Recording process stopped, but no audio data was written to disk.",
        }

    duration = _compute_wav_duration_safe(audio_path, elapsed)
    response: dict[str, Any] = {
        "status": "stopped",
        "path": str(audio_path),
        "duration": duration,
        "backend": backend,
        "message": f"Stopped voice recording ({duration}s saved to {audio_path}).",
    }

    if auto_transcribe:
        try:
            tx = transcribe_file(
                str(audio_path),
                config=config,
                cleanup_source_on_success=cleanup_on_transcribe_success,
            )
            response["transcript"] = tx.get("text", "")
            response["language"] = tx.get("language", "en")
            response["model"] = tx.get("model", "")
            response["cleaned_up"] = bool(cleanup_on_transcribe_success and not audio_path.exists())
        except (VoiceInputError, VoiceBackendError) as exc:
            response["transcription_error"] = str(exc)
            response["preserved_audio_path"] = str(audio_path)

    return response
