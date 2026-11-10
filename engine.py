"""Core audio validation, model management, and offline transcription engine."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".wav",
    ".mp3",
    ".ogg",
    ".opus",
    ".m4a",
    ".flac",
    ".webm",
    ".aac",
    ".mp4",
    ".wma",
    ".caf",
})

MODEL_DISK_MB: dict[str, int] = {
    "tiny.en": 75,
    "tiny": 75,
    "base.en": 145,
    "base": 145,
    "small.en": 466,
    "small": 466,
    "medium.en": 1500,
    "medium": 1500,
    "large-v3-turbo": 1600,
    "large-v3": 3100,
}

HF_REPO_ALIASES: dict[str, tuple[str, ...]] = {
    "tiny.en": ("Systran/faster-whisper-tiny.en",),
    "tiny": ("Systran/faster-whisper-tiny",),
    "base.en": ("Systran/faster-whisper-base.en",),
    "base": ("Systran/faster-whisper-base",),
    "small.en": ("Systran/faster-whisper-small.en",),
    "small": ("Systran/faster-whisper-small",),
    "medium.en": ("Systran/faster-whisper-medium.en",),
    "medium": ("Systran/faster-whisper-medium",),
    "large-v3-turbo": (
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "deepdml/faster-whisper-large-v3-turbo-ct2",
    ),
    "large-v3": ("Systran/faster-whisper-large-v3",),
}

_MODEL_CACHE_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_INPROC_MODELS: dict[tuple[str, str, str], Any] = {}
_LAST_TRANSCRIPT_MEM: dict[str, Any] | None = None


class VoiceInputError(ValueError):
    """Raised when an input audio file fails validation."""

    def __init__(self, message: str, *, code: str = "invalid_input", temp_path: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.temp_path = temp_path


class VoiceBackendError(RuntimeError):
    """Raised when no local transcription backend or model is available."""

    def __init__(self, message: str, *, code: str = "backend_unavailable", temp_path: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.temp_path = temp_path


def get_hermes_home() -> Path:
    """Resolve HERMES_HOME dynamically without hardcoded user paths."""
    env_home = os.environ.get("HERMES_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()
    return (Path.home() / ".hermes").resolve()


def get_plugin_data_dir() -> Path:
    """Return the persistent data directory for hermes-local-voice under HERMES_HOME."""
    override = os.environ.get("HERMES_VOICE_DATA_DIR", "").strip()
    if override:
        base = Path(override).expanduser().resolve()
    else:
        base = get_hermes_home() / "plugin-data" / "hermes-local-voice"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_model_dir(config: dict[str, Any] | None = None) -> Path:
    """Return the model storage directory under HERMES_HOME unless overridden."""
    cfg = config or {}
    custom = str(cfg.get("model_dir") or os.environ.get("HERMES_VOICE_MODEL_DIR") or "").strip()
    if custom:
        target = Path(custom).expanduser().resolve()
    else:
        target = get_plugin_data_dir() / "models"
    target.mkdir(parents=True, exist_ok=True)
    return target


def find_binary(name: str) -> str | None:
    """Locate an executable on PATH or common user bin locations."""
    found = shutil.which(name)
    if found:
        return found
    candidates = [
        Path.home() / "bin" / name,
        Path.home() / ".local" / "bin" / name,
        Path("/usr/bin") / name,
        Path("/usr/local/bin") / name,
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def find_whisper_python() -> str | None:
    """Find a Python interpreter that has faster_whisper installed."""
    env_py = os.environ.get("HERMES_VOICE_PYTHON", "").strip()
    candidates: list[Path] = []
    if env_py:
        candidates.append(Path(env_py).expanduser())
    hermes_home = get_hermes_home()
    candidates.extend([
        hermes_home / "venvs" / "whisper" / "bin" / "python",
        Path.home() / ".venvs" / "whisper" / "bin" / "python",
    ])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _looks_like_valid_container_magic(header: bytes, ext: str) -> bool:
    """Verify container magic bytes for common audio formats."""
    if len(header) < 4:
        return False
    if ext == ".wav":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    if ext in (".ogg", ".opus"):
        return header[:4] == b"OggS"
    if ext == ".flac":
        return header[:4] == b"fLaC"
    if ext == ".mp3":
        if header[:3] == b"ID3":
            return True
        return len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0
    if ext in (".m4a", ".mp4", ".aac"):
        if len(header) >= 8 and header[4:8] == b"ftyp":
            return True
        return len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xF0) == 0xF0
    if ext == ".webm":
        return header[:4] == b"\x1a\x45\xdf\xa3"
    if ext == ".caf":
        return header[:4] == b"caff"
    if ext == ".wma":
        return header[:4] == b"\x30\x26\xb2\x75"
    return True


def _probe_wav_duration(path: Path) -> float | None:
    """Inspect a PCM WAV file using the standard library wave module."""
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            if rate <= 0 or channels <= 0:
                raise VoiceInputError(
                    f"Corrupt WAV header in {path}: invalid sample rate ({rate}) or channels ({channels}).",
                    code="corrupt_file",
                )
            if frames <= 0:
                raise VoiceInputError(
                    f"Audio file contains zero audio frames (0.0s duration): {path}.",
                    code="zero_length_audio",
                )
            return round(frames / float(rate), 3)
    except VoiceInputError:
        raise
    except wave.Error as exc:
        return None


def _probe_with_ffprobe(path: Path) -> float | None:
    """Probe duration and audio stream validity via ffprobe when available."""
    ffprobe_bin = find_binary("ffprobe")
    if not ffprobe_bin:
        return None
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VoiceInputError(
            f"Failed to probe audio file {path}: {exc}",
            code="corrupt_file",
        ) from exc

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip() or "unrecognized audio container"
        raise VoiceInputError(
            f"Corrupt or unreadable audio file {path}: {stderr}",
            code="corrupt_file",
        )

    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise VoiceInputError(
            f"Corrupt audio metadata in {path}.",
            code="corrupt_file",
        ) from exc

    streams = payload.get("streams") or []
    if not streams:
        raise VoiceInputError(
            f"File contains no decodable audio stream: {path}.",
            code="corrupt_file",
        )

    duration_raw = (
        streams[0].get("duration")
        or (payload.get("format") or {}).get("duration")
    )
    if duration_raw is not None:
        try:
            dur = float(duration_raw)
            if dur <= 0.0:
                raise VoiceInputError(
                    f"Audio file has zero duration (0.0s): {path}.",
                    code="zero_length_audio",
                )
            return round(dur, 3)
        except ValueError:
            pass
    return None


def validate_audio_file(path_str: str, max_file_mb: float = 100.0) -> tuple[Path, float]:
    """Validate an audio file path and return (resolved_path, duration_seconds).

    Raises VoiceInputError with a specific code and message on invalid input:
    - missing_file
    - is_directory
    - unsupported_extension
    - empty_file
    - oversized_file
    - corrupt_file
    - zero_length_audio
    """
    if not path_str or not str(path_str).strip():
        raise VoiceInputError(
            "Audio file path is required and cannot be empty.",
            code="missing_file",
        )

    path = Path(str(path_str).strip()).expanduser()
    if not path.exists():
        raise VoiceInputError(
            f"Audio file not found: {path}. Verify the path exists before transcribing.",
            code="missing_file",
        )
    if path.is_dir():
        raise VoiceInputError(
            f"Path is a directory, not an audio file: {path}.",
            code="is_directory",
        )

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise VoiceInputError(
            f"Unsupported file extension '{ext or '(none)'}' for {path.name}. Supported extensions: {allowed}.",
            code="unsupported_extension",
        )

    try:
        stat = path.stat()
    except OSError as exc:
        raise VoiceInputError(
            f"Cannot stat audio file {path}: {exc}",
            code="missing_file",
        ) from exc

    if stat.st_size == 0:
        raise VoiceInputError(
            f"Audio file is empty (0 bytes): {path}.",
            code="empty_file",
        )

    max_bytes = int(float(max_file_mb) * 1024 * 1024)
    if stat.st_size > max_bytes:
        actual_mb = round(stat.st_size / (1024 * 1024), 2)
        raise VoiceInputError(
            f"Audio file size ({actual_mb} MB) exceeds maximum allowed limit ({max_file_mb} MB): {path}.",
            code="oversized_file",
        )

    try:
        with path.open("rb") as fh:
            header = fh.read(64)
    except OSError as exc:
        raise VoiceInputError(
            f"Cannot read audio file {path}: {exc}",
            code="corrupt_file",
        ) from exc

    if not _looks_like_valid_container_magic(header, ext):
        raise VoiceInputError(
            f"Corrupt or invalid {ext} audio header in {path}.",
            code="corrupt_file",
        )

    if ext == ".wav":
        wav_dur = _probe_wav_duration(path)
        if wav_dur is not None:
            return path.resolve(), wav_dur

    probed_dur = _probe_with_ffprobe(path)
    if probed_dur is not None:
        return path.resolve(), probed_dur

    if ext == ".wav":
        raise VoiceInputError(
            f"Corrupt or unreadable WAV structure in {path}.",
            code="corrupt_file",
        )

    return path.resolve(), 0.0


def convert_to_wav_16k(source_path: Path) -> tuple[Path, bool]:
    """Convert non-WAV or non-16kHz audio to a temporary 16 kHz mono PCM WAV.

    Returns (audio_path_to_use, is_temporary).
    If source_path is already a valid PCM WAV, returns (source_path, False).
    """
    if source_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(source_path), "rb") as wf:
                if wf.getnchannels() == 1 and wf.getframerate() == 16000 and wf.getsampwidth() == 2:
                    return source_path, False
        except wave.Error:
            pass

    ffmpeg_bin = find_binary("ffmpeg")
    if not ffmpeg_bin:
        if source_path.suffix.lower() == ".wav":
            return source_path, False
        raise VoiceBackendError(
            f"ffmpeg is required to convert '{source_path.suffix}' files to 16 kHz WAV, "
            "but ffmpeg was not found on PATH.",
            code="ffmpeg_missing",
        )

    temp_dir = get_plugin_data_dir() / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_wav = temp_dir / f"converted_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        str(source_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(temp_wav),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if temp_wav.exists():
            temp_wav.unlink(missing_ok=True)
        raise VoiceInputError(
            f"Audio conversion failed for {source_path}: {exc}",
            code="corrupt_file",
        ) from exc

    if proc.returncode != 0 or not temp_wav.exists() or temp_wav.stat().st_size == 0:
        stderr = (proc.stderr or "").strip() or "ffmpeg conversion failed"
        if temp_wav.exists():
            temp_wav.unlink(missing_ok=True)
        raise VoiceInputError(
            f"Corrupt or undecodable audio in {source_path}: {stderr}",
            code="corrupt_file",
        )

    return temp_wav, True


def find_cached_model_path(model_size: str, config: dict[str, Any] | None = None) -> Path | None:
    """Locate a downloaded model on disk in HERMES_HOME or the local HuggingFace cache."""
    model_dir = get_model_dir(config)
    direct_dir = model_dir / model_size
    if (direct_dir / "model.bin").is_file():
        return direct_dir

    aliases = list(HF_REPO_ALIASES.get(model_size, ()))
    aliases.append(f"Systran/faster-whisper-{model_size}")

    search_roots = [
        model_dir,
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        search_roots.insert(1, Path(hf_home).expanduser() / "hub")

    for root in search_roots:
        if not root.is_dir():
            continue
        for repo_id in aliases:
            folder_name = "models--" + repo_id.replace("/", "--")
            snapshots_dir = root / folder_name / "snapshots"
            if snapshots_dir.is_dir():
                for snap in sorted(snapshots_dir.iterdir(), reverse=True):
                    if (snap / "model.bin").is_file():
                        return snap
    return None


def list_models_status(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return all supported Whisper models with disk cost in MB and local cache status."""
    rows: list[dict[str, Any]] = []
    for name, size_mb in MODEL_DISK_MB.items():
        cached_path = find_cached_model_path(name, config)
        rows.append({
            "model": name,
            "approx_size_mb": size_mb,
            "cached": cached_path is not None,
            "path": str(cached_path) if cached_path else None,
        })
    return rows


def download_model(model_size: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Explicitly download a Whisper model into HERMES_HOME/plugin-data/hermes-local-voice/models.

    Reports approximate disk cost beforehand and fails cleanly when offline.
    """
    if model_size not in MODEL_DISK_MB:
        supported = ", ".join(MODEL_DISK_MB.keys())
        raise VoiceInputError(
            f"Unknown model_size '{model_size}'. Supported models: {supported}.",
            code="unknown_model",
        )

    target_root = get_model_dir(config)
    approx_mb = MODEL_DISK_MB[model_size]
    existing = find_cached_model_path(model_size, config)
    if existing is not None:
        return {
            "status": "already_cached",
            "model": model_size,
            "approx_size_mb": approx_mb,
            "model_dir": str(existing),
            "message": f"Model '{model_size}' (~{approx_mb} MB) is already cached at {existing}.",
        }

    py_bin = find_whisper_python()
    script = (
        "import sys\n"
        "from faster_whisper.utils import download_model\n"
        "path = download_model(sys.argv[1], output_dir=sys.argv[2], local_files_only=False)\n"
        "print(path)\n"
    )
    out_dir = target_root / model_size
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if py_bin:
            proc = subprocess.run(
                [py_bin, "-c", script, model_size, str(out_dir)],
                capture_output=True,
                text=True,
                timeout=600,
                stdin=subprocess.DEVNULL,
            )
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip() or "model download failed"
                shutil.rmtree(out_dir, ignore_errors=True)
                raise VoiceBackendError(
                    f"Could not download model '{model_size}' (~{approx_mb} MB) to {out_dir}. "
                    f"Check network connectivity. Details: {stderr}",
                    code="download_failed",
                )
        else:
            from faster_whisper.utils import download_model as fw_download  # type: ignore[import-not-found]
            fw_download(model_size, output_dir=str(out_dir), local_files_only=False)
    except KeyboardInterrupt:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    except ImportError as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise VoiceBackendError(
            f"faster-whisper is not installed. Cannot download model '{model_size}'.",
            code="backend_unavailable",
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise VoiceBackendError(
            f"Network or subprocess failure while downloading '{model_size}' (~{approx_mb} MB): {exc}",
            code="download_failed",
        ) from exc

    return {
        "status": "downloaded",
        "model": model_size,
        "approx_size_mb": approx_mb,
        "model_dir": str(out_dir),
        "message": f"Downloaded model '{model_size}' (~{approx_mb} MB) to {out_dir}.",
    }


def _is_loopback_url(url: str) -> bool:
    """Ensure server_url points strictly to a local loopback address."""
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1")


def _try_loopback_server(audio_path: Path, server_url: str, timeout: float = 30.0) -> dict[str, Any] | None:
    """Attempt transcription via the local faster-whisper HTTP job server on loopback."""
    if not server_url or not _is_loopback_url(server_url):
        return None

    endpoint = server_url.rstrip("/") + "/transcribe"
    boundary = f"----HermesLocalVoice{uuid.uuid4().hex}"
    try:
        file_bytes = audio_path.read_bytes()
    except OSError:
        return None

    header_part = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8")
    footer_part = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header_part + file_bytes + footer_part

    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None

    text = str(payload.get("text") or payload.get("raw_text") or "").strip()
    return {
        "text": text,
        "duration": payload.get("duration"),
        "language": payload.get("language"),
        "model": payload.get("model") or "large-v3-turbo (loopback-server)",
        "backend": "loopback_server",
    }


def _transcribe_with_faster_whisper(
    audio_path: Path,
    model_size: str,
    compute_type: str,
    language: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run offline faster-whisper transcription in-process or via local whisper venv."""
    cached_model_path = find_cached_model_path(model_size, config)
    if cached_model_path is None:
        fallback_order = ["large-v3-turbo", "base.en", "small.en", "tiny.en"]
        for alt in fallback_order:
            alt_path = find_cached_model_path(alt, config)
            if alt_path is not None:
                model_size = alt
                cached_model_path = alt_path
                break

    if cached_model_path is None:
        approx_mb = MODEL_DISK_MB.get(model_size, 145)
        model_dir = get_model_dir(config)
        raise VoiceBackendError(
            f"Offline mode: Whisper model '{model_size}' (~{approx_mb} MB) is not cached locally in "
            f"{model_dir}. Run 'hermes voice --download {model_size}' once while online to fetch it.",
            code="model_not_cached",
        )

    lang_arg = None if not language or language.lower() == "auto" else language

    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        has_inproc = True
    except ImportError:
        has_inproc = False

    if has_inproc:
        cache_key = (str(cached_model_path), compute_type, model_size)
        with _MODEL_CACHE_LOCK:
            model = _INPROC_MODELS.get(cache_key)
            if model is None:
                model = WhisperModel(
                    str(cached_model_path),
                    device="cpu",
                    compute_type=compute_type,
                    local_files_only=True,
                )
                _INPROC_MODELS[cache_key] = model
            segments, info = model.transcribe(
                str(audio_path),
                language=lang_arg,
                beam_size=5,
                vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            detected_lang = getattr(info, "language", None) or (lang_arg or "en")
            info_duration = getattr(info, "duration", None)
        return {
            "text": text,
            "duration": round(float(info_duration), 3) if info_duration else None,
            "language": detected_lang,
            "model": model_size,
            "backend": "faster_whisper_inproc",
        }

    py_bin = find_whisper_python()
    if not py_bin:
        raise VoiceBackendError(
            "faster-whisper is not installed in the active Python runtime and no local whisper "
            "virtualenv was found. Install faster-whisper or configure server_url.",
            code="backend_unavailable",
        )

    worker_code = (
        "import json, sys\n"
        "from faster_whisper import WhisperModel\n"
        "model_path, audio_file, compute_type, lang = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]\n"
        "lang_arg = None if lang in ('', 'auto') else lang\n"
        "model = WhisperModel(model_path, device='cpu', compute_type=compute_type, local_files_only=True)\n"
        "segments, info = model.transcribe(audio_file, language=lang_arg, beam_size=5, vad_filter=True)\n"
        "text = ' '.join(s.text.strip() for s in segments).strip()\n"
        "print(json.dumps({\n"
        "    'text': text,\n"
        "    'duration': round(float(info.duration), 3) if getattr(info, 'duration', None) else None,\n"
        "    'language': getattr(info, 'language', None) or (lang_arg or 'en'),\n"
        "}))\n"
    )
    with _MODEL_CACHE_LOCK:
        proc = subprocess.run(
            [py_bin, "-c", worker_code, str(cached_model_path), str(audio_path), compute_type, language or "en"],
            capture_output=True,
            text=True,
            timeout=180,
            stdin=subprocess.DEVNULL,
        )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip() or "faster-whisper subprocess failed"
        raise VoiceBackendError(
            f"Local faster-whisper transcription failed: {stderr}",
            code="transcription_failed",
        )
    try:
        out = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise VoiceBackendError(
            f"Malformed output from local faster-whisper worker: {proc.stdout!r}",
            code="transcription_failed",
        ) from exc

    return {
        "text": str(out.get("text") or "").strip(),
        "duration": out.get("duration"),
        "language": out.get("language") or (lang_arg or "en"),
        "model": model_size,
        "backend": "faster_whisper_venv",
    }


def _save_last_transcript(entry: dict[str, Any]) -> None:
    """Persist the most recent transcript atomically in memory and under HERMES_HOME."""
    global _LAST_TRANSCRIPT_MEM
    with _STATE_LOCK:
        _LAST_TRANSCRIPT_MEM = dict(entry)
        state_file = get_plugin_data_dir() / "last_transcript.json"
        tmp_file = state_file.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp_file.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_file, state_file)
        except OSError:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)


def get_last_transcript() -> dict[str, Any] | None:
    """Retrieve the most recent transcript from memory or persistent storage."""
    with _STATE_LOCK:
        if _LAST_TRANSCRIPT_MEM is not None:
            return dict(_LAST_TRANSCRIPT_MEM)
        state_file = get_plugin_data_dir() / "last_transcript.json"
        if not state_file.is_file():
            return None
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


def clear_last_transcript() -> None:
    """Clear stored transcript state (used by test fixtures)."""
    global _LAST_TRANSCRIPT_MEM
    with _STATE_LOCK:
        _LAST_TRANSCRIPT_MEM = None
        state_file = get_plugin_data_dir() / "last_transcript.json"
        if state_file.exists():
            state_file.unlink(missing_ok=True)


def transcribe_file(
    path_str: str,
    config: dict[str, Any] | None = None,
    *,
    cleanup_source_on_success: bool = False,
) -> dict[str, Any]:
    """Validate and transcribe an audio file offline.

    Temporary converted files (and captured recordings when cleanup_source_on_success=True)
    are removed after transcription succeeds, and preserved with their path reported
    when transcription fails so the user never loses a recording.
    """
    cfg = config or {}
    max_file_mb = float(cfg.get("max_file_mb") or 100)
    model_size = str(cfg.get("model_size") or "base.en").strip()
    compute_type = str(cfg.get("compute_type") or "int8").strip()
    language = str(cfg.get("language") or "en").strip()
    server_url = str(cfg.get("server_url") if "server_url" in cfg else "http://127.0.0.1:8001").strip()
    prefer_server = bool(cfg.get("prefer_server", True))

    validated_path, probed_duration = validate_audio_file(path_str, max_file_mb=max_file_mb)
    work_path, is_temp_conversion = convert_to_wav_16k(validated_path)

    if probed_duration <= 0.0 and work_path.suffix.lower() == ".wav":
        wav_dur = _probe_wav_duration(work_path)
        if wav_dur is not None:
            probed_duration = wav_dur

    result: dict[str, Any] | None = None
    try:
        if prefer_server and server_url:
            result = _try_loopback_server(work_path, server_url=server_url)

        if result is None:
            result = _transcribe_with_faster_whisper(
                work_path,
                model_size=model_size,
                compute_type=compute_type,
                language=language,
                config=cfg,
            )
    except Exception as exc:
        preserved_path = str(validated_path)
        if is_temp_conversion and work_path.exists():
            preserved_path = str(work_path)
        if isinstance(exc, (VoiceInputError, VoiceBackendError)):
            exc.temp_path = preserved_path
            raise
        raise VoiceBackendError(
            f"Transcription failed for {validated_path}: {exc}. Audio preserved at {preserved_path}.",
            code="transcription_failed",
            temp_path=preserved_path,
        ) from exc

    if is_temp_conversion and work_path.exists():
        work_path.unlink(missing_ok=True)
    if cleanup_source_on_success and validated_path.exists():
        validated_path.unlink(missing_ok=True)

    final_duration = result.get("duration")
    if final_duration is None or float(final_duration) <= 0.0:
        final_duration = probed_duration
    final_duration = round(float(final_duration or 0.0), 3)

    detected_lang = str(result.get("language") or (language if language != "auto" else "en"))
    used_model = str(result.get("model") or model_size)

    output = {
        "status": "ok",
        "text": str(result.get("text") or "").strip(),
        "duration": final_duration,
        "language": detected_lang,
        "model": used_model,
        "backend": str(result.get("backend") or "local"),
        "path": str(validated_path),
        "timestamp": round(time.time(), 3),
    }
    _save_last_transcript(output)
    return output
