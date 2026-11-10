"""Unit, integration, and concurrency test suite for hermes-local-voice."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import struct
from typing import Any
from unittest.mock import patch
import wave

import pytest

import hermes_local_voice as plugin_root
from hermes_local_voice.cli import handle_slash_voice, run_voice_cli, setup_voice_cli
from hermes_local_voice.engine import (
    MODEL_DISK_MB,
    VoiceBackendError,
    VoiceInputError,
    clear_last_transcript,
    download_model,
    find_cached_model_path,
    find_whisper_python,
    get_hermes_home,
    get_last_transcript,
    get_model_dir,
    list_models_status,
    transcribe_file,
    validate_audio_file,
    _try_loopback_server,
)
from hermes_local_voice.hooks import (
    extract_audio_paths_from_message,
    on_pre_gateway_dispatch,
    on_pre_llm_call,
)
from hermes_local_voice.recorder import start_recording, stop_recording
from hermes_local_voice.tools import (
    handle_transcribe_audio,
    handle_transcribe_last,
    handle_voice_record_start,
    handle_voice_record_stop,
    set_plugin_config,
)

FIXTURE_WAV = Path(__file__).resolve().parent / "fixtures" / "sample.wav"


def _write_valid_pcm_wav(path: Path, duration_sec: float = 0.5, sample_rate: int = 16000) -> Path:
    """Create a valid 16 kHz mono 16-bit PCM WAV file of the given duration."""
    num_frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return path


@pytest.fixture(autouse=True)
def isolated_plugin_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate HERMES_VOICE_DATA_DIR per test so tests never pollute real user state."""
    data_dir = tmp_path / "plugin_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_VOICE_DATA_DIR", str(data_dir))
    clear_last_transcript()
    set_plugin_config({
        "model_size": "base.en",
        "compute_type": "int8",
        "language": "en",
        "model_dir": "",
        "max_file_mb": 100,
        "server_url": "http://127.0.0.1:8001",
        "prefer_server": True,
        "auto_transcribe_inbound": True,
    })
    yield data_dir
    clear_last_transcript()


# ---------------------------------------------------------------------------
# 1. Input Validation Unit Tests
# ---------------------------------------------------------------------------


def test_validate_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.wav"
    with pytest.raises(VoiceInputError) as exc_info:
        validate_audio_file(str(missing))
    assert exc_info.value.code == "missing_file"
    assert "not found" in str(exc_info.value).lower()


def test_validate_empty_path() -> None:
    with pytest.raises(VoiceInputError) as exc_info:
        validate_audio_file("   ")
    assert exc_info.value.code == "missing_file"


def test_validate_directory_rejected(tmp_path: Path) -> None:
    with pytest.raises(VoiceInputError) as exc_info:
        validate_audio_file(str(tmp_path))
    assert exc_info.value.code == "is_directory"


def test_validate_wrong_extension_rejected(tmp_path: Path) -> None:
    bad_ext = tmp_path / "notes.txt"
    bad_ext.write_text("not an audio file", encoding="utf-8")
    with pytest.raises(VoiceInputError) as exc_info:
        validate_audio_file(str(bad_ext))
    assert exc_info.value.code == "unsupported_extension"
    assert ".txt" in str(exc_info.value)


def test_validate_zero_byte_file_rejected(tmp_path: Path) -> None:
    empty_wav = tmp_path / "empty.wav"
    empty_wav.write_bytes(b"")
    with pytest.raises(VoiceInputError) as exc_info:
        validate_audio_file(str(empty_wav))
    assert exc_info.value.code == "empty_file"
    assert "0 bytes" in str(exc_info.value)


def test_validate_zero_length_audio_frames_rejected(tmp_path: Path) -> None:
    zero_frames_wav = tmp_path / "zero_frames.wav"
    _write_valid_pcm_wav(zero_frames_wav, duration_sec=0.0)
    with pytest.raises(VoiceInputError) as exc_info:
        validate_audio_file(str(zero_frames_wav))
    assert exc_info.value.code == "zero_length_audio"


def test_validate_oversized_file_rejected(tmp_path: Path) -> None:
    big_wav = tmp_path / "oversized.wav"
    _write_valid_pcm_wav(big_wav, duration_sec=1.0)
    with pytest.raises(VoiceInputError) as exc_info:
        validate_audio_file(str(big_wav), max_file_mb=0.001)
    assert exc_info.value.code == "oversized_file"
    assert "exceeds maximum allowed limit" in str(exc_info.value)


def test_validate_corrupt_wav_header_rejected(tmp_path: Path) -> None:
    corrupt_wav = tmp_path / "corrupt.wav"
    corrupt_wav.write_bytes(b"NOT_A_RIFF_HEADER_AT_ALL_1234567890")
    with pytest.raises(VoiceInputError) as exc_info:
        validate_audio_file(str(corrupt_wav))
    assert exc_info.value.code == "corrupt_file"


def test_validate_corrupt_ogg_and_mp3_rejected(tmp_path: Path) -> None:
    corrupt_ogg = tmp_path / "bad.ogg"
    corrupt_ogg.write_bytes(b"RANDOM_BYTES_NOT_OGGS_CONTAINER")
    with pytest.raises(VoiceInputError) as exc_info:
        validate_audio_file(str(corrupt_ogg))
    assert exc_info.value.code == "corrupt_file"

    corrupt_mp3 = tmp_path / "bad.mp3"
    corrupt_mp3.write_bytes(b"\x00\x00\x00\x00NOT_ID3_OR_SYNC")
    with pytest.raises(VoiceInputError) as exc_info:
        validate_audio_file(str(corrupt_mp3))
    assert exc_info.value.code == "corrupt_file"


def test_tool_handler_returns_structured_validation_error(tmp_path: Path) -> None:
    raw = handle_transcribe_audio({"path": str(tmp_path / "missing_clip.wav")})
    payload = json.loads(raw)
    assert payload["status"] == "error"
    assert payload["error_code"] == "missing_file"


# ---------------------------------------------------------------------------
# 2. Temp File Cleanup and Preservation on Failure Tests
# ---------------------------------------------------------------------------


def test_audio_preserved_when_transcription_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = _write_valid_pcm_wav(tmp_path / "important_recording.wav", duration_sec=0.5)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty_hf"))
    cfg = {
        "model_size": "tiny.en",
        "model_dir": str(tmp_path / "empty_models"),
        "prefer_server": False,
        "server_url": "",
    }
    set_plugin_config(cfg)

    with patch("hermes_local_voice.engine.find_cached_model_path", return_value=None):
        raw = handle_transcribe_audio({
            "path": str(clip),
            "cleanup_on_success": True,
        })
    payload = json.loads(raw)
    assert payload["status"] == "error"
    assert payload["error_code"] == "model_not_cached"
    assert payload.get("preserved_audio_path") == str(clip.resolve())
    assert clip.exists(), "Recording must be preserved on disk when transcription fails"


def test_temp_recording_cleaned_up_on_success(tmp_path: Path) -> None:
    clip = _write_valid_pcm_wav(tmp_path / "temp_capture.wav", duration_sec=0.5)
    fake_server_resp = {
        "text": "Cleaned up after success.",
        "duration": 0.5,
        "language": "en",
        "model": "base.en",
        "backend": "loopback_server",
    }
    with patch("hermes_local_voice.engine._try_loopback_server", return_value=fake_server_resp):
        res = transcribe_file(str(clip), cleanup_source_on_success=True)
    assert res["status"] == "ok"
    assert res["text"] == "Cleaned up after success."
    assert not clip.exists(), "Temporary source file should be removed after successful transcription"


# ---------------------------------------------------------------------------
# 3. Model Management & Offline-by-Default Tests
# ---------------------------------------------------------------------------


def test_hermes_home_and_model_dir_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom_home = tmp_path / "custom_hermes_home"
    monkeypatch.setenv("HERMES_HOME", str(custom_home))
    monkeypatch.delenv("HERMES_VOICE_DATA_DIR", raising=False)
    monkeypatch.delenv("HERMES_VOICE_MODEL_DIR", raising=False)

    assert get_hermes_home() == custom_home.resolve()
    expected_models = custom_home.resolve() / "plugin-data" / "hermes-local-voice" / "models"
    assert get_model_dir() == expected_models
    assert expected_models.is_dir()


def test_models_status_reports_disk_cost(tmp_path: Path) -> None:
    rows = list_models_status({"model_dir": str(tmp_path / "models")})
    by_name = {r["model"]: r for r in rows}
    assert "base.en" in by_name
    assert by_name["base.en"]["approx_size_mb"] == MODEL_DISK_MB["base.en"] == 145
    assert "small.en" in by_name
    assert by_name["small.en"]["approx_size_mb"] == 466


def test_download_model_rejects_unknown_model() -> None:
    with pytest.raises(VoiceInputError) as exc_info:
        download_model("nonexistent-whisper-99b")
    assert exc_info.value.code == "unknown_model"


# ---------------------------------------------------------------------------
# 4. Microphone Recording & transcribe_last() Unit Tests
# ---------------------------------------------------------------------------


def test_voice_record_start_degrades_cleanly_without_mic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_VOICE_DISABLE_MIC", "1")
    raw = handle_voice_record_start({})
    payload = json.loads(raw)
    assert payload["status"] == "unavailable"
    assert payload["error_code"] == "microphone_unavailable"
    assert "transcribe_audio" in payload["message"]


def test_voice_record_stop_when_not_recording() -> None:
    raw = handle_voice_record_stop({})
    payload = json.loads(raw)
    assert payload["status"] == "error"
    assert payload["error_code"] == "not_recording"


def test_voice_record_lifecycle_with_simulated_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HERMES_VOICE_DISABLE_MIC", raising=False)

    class DummyProc:
        pid = 999991

        def poll(self) -> None:
            return None

    def fake_detect(out_path: Path) -> tuple[list[str], str]:
        _write_valid_pcm_wav(out_path, duration_sec=1.25)
        return (["pw-record", str(out_path)], "pw-record")

    with (
        patch("hermes_local_voice.recorder.detect_capture_command", side_effect=fake_detect),
        patch("hermes_local_voice.recorder.subprocess.Popen", return_value=DummyProc()),
        patch("hermes_local_voice.recorder.time.sleep"),
        patch("hermes_local_voice.recorder._pid_is_alive", return_value=False),
    ):
        start_res = start_recording()
        assert start_res["status"] == "recording"
        assert start_res["pid"] == 999991
        recorded_path = Path(start_res["path"])
        assert recorded_path.is_file()

        stop_res = stop_recording(auto_transcribe=False)
        assert stop_res["status"] == "stopped"
        assert stop_res["path"] == str(recorded_path)
        assert abs(stop_res["duration"] - 1.25) < 0.05


def test_transcribe_last_lifecycle(tmp_path: Path) -> None:
    empty_res = json.loads(handle_transcribe_last({}))
    assert empty_res["status"] == "empty"

    clip = _write_valid_pcm_wav(tmp_path / "sample_last.wav", duration_sec=0.75)
    fake_resp = {
        "text": "Testing transcribe last state persistence.",
        "duration": 0.75,
        "language": "en",
        "model": "base.en",
        "backend": "loopback_server",
    }
    with patch("hermes_local_voice.engine._try_loopback_server", return_value=fake_resp):
        tx_res = json.loads(handle_transcribe_audio({"path": str(clip)}))
    assert tx_res["status"] == "ok"

    last_res = json.loads(handle_transcribe_last({}))
    assert last_res["status"] == "ok"
    assert last_res["text"] == "Testing transcribe last state persistence."
    assert last_res["duration"] == 0.75
    assert last_res["language"] == "en"
    assert last_res["model"] == "base.en"


# ---------------------------------------------------------------------------
# 5. Message Hooks & CLI Subcommand Unit Tests
# ---------------------------------------------------------------------------


def test_pre_gateway_dispatch_hook_rewrites_audio_attachment(tmp_path: Path) -> None:
    clip = _write_valid_pcm_wav(tmp_path / "voice_note.wav", duration_sec=1.5)
    event = {
        "text": "Please summarize this note",
        "media_urls": [str(clip)],
        "media_types": ["audio/wav"],
    }
    fake_resp = {
        "text": "Deploy the staging build at five PM.",
        "duration": 1.5,
        "language": "en",
        "model": "base.en",
        "backend": "loopback_server",
    }
    with patch("hermes_local_voice.engine._try_loopback_server", return_value=fake_resp):
        hook_out = on_pre_gateway_dispatch(event)

    assert hook_out is not None
    assert hook_out["action"] == "rewrite"
    assert "Deploy the staging build at five PM." in hook_out["text"]
    assert "Please summarize this note" in hook_out["text"]


def test_pre_llm_call_hook_injects_transcript_context(tmp_path: Path) -> None:
    clip = _write_valid_pcm_wav(tmp_path / "chat_clip.wav", duration_sec=2.0)
    msg = f"Can you check what I said in @{clip}?"
    paths = extract_audio_paths_from_message(msg)
    assert paths == [clip.resolve()]

    fake_resp = {
        "text": "Check the soil moisture sensors in block four.",
        "duration": 2.0,
        "language": "en",
        "model": "base.en",
        "backend": "loopback_server",
    }
    with patch("hermes_local_voice.engine._try_loopback_server", return_value=fake_resp):
        hook_out = on_pre_llm_call(session_id="sess-1", user_message=msg)

    assert hook_out is not None
    assert "context" in hook_out
    assert "Check the soil moisture sensors in block four." in hook_out["context"]


def test_cli_subcommand_and_slash_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    clip = _write_valid_pcm_wav(tmp_path / "cli_clip.wav", duration_sec=1.0)
    parser = argparse.ArgumentParser(prog="hermes voice")
    setup_voice_cli(parser)

    fake_resp = {
        "text": "Headless CLI transcript output works.",
        "duration": 1.0,
        "language": "en",
        "model": "base.en",
        "backend": "loopback_server",
    }
    with patch("hermes_local_voice.engine._try_loopback_server", return_value=fake_resp):
        args = parser.parse_args([str(clip)])
        code = run_voice_cli(args)
        assert code == 0
        captured = capsys.readouterr()
        assert captured.out.strip() == "Headless CLI transcript output works."

        slash_out = handle_slash_voice("last")
        assert "Headless CLI transcript output works." in slash_out


def test_plugin_register_wires_expected_capabilities() -> None:
    recorded: dict[str, list[str]] = {"tools": [], "hooks": [], "commands": []}

    class StubContext:
        def get_config(self, key: str, default: Any = None) -> Any:
            return default

        def register_tool(self, name: str, **kwargs: Any) -> None:
            recorded["tools"].append(name)

        def register_hook(self, hook_name: str, callback: Any, **kwargs: Any) -> None:
            recorded["hooks"].append(hook_name)

        def register_cli_command(self, name: str, **kwargs: Any) -> None:
            recorded["commands"].append(f"cli:{name}")

        def register_command(self, name: str, **kwargs: Any) -> None:
            recorded["commands"].append(f"slash:{name}")

    plugin_root.register(StubContext())
    assert recorded["tools"] == [
        "transcribe_audio",
        "voice_record_start",
        "voice_record_stop",
        "transcribe_last",
    ]
    assert recorded["hooks"] == ["pre_gateway_dispatch", "pre_llm_call"]
    assert "cli:voice" in recorded["commands"]
    assert "slash:local-voice" in recorded["commands"]


# ---------------------------------------------------------------------------
# 6. Integration Test (Real Fixture Clip Transcription)
# ---------------------------------------------------------------------------


def _has_real_whisper_backend() -> bool:
    if _try_loopback_server(FIXTURE_WAV, "http://127.0.0.1:8001", timeout=5.0) is not None:
        return True
    any_cached = any(
        find_cached_model_path(m) is not None
        for m in ("large-v3-turbo", "base.en", "small.en", "tiny.en")
    )
    if not any_cached:
        return False
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return find_whisper_python() is not None


@pytest.mark.skipif(
    not FIXTURE_WAV.is_file() or not _has_real_whisper_backend(),
    reason="Skipping integration test: sample.wav or local Whisper model/backend is absent.",
)
def test_integration_transcribe_real_fixture_clip() -> None:
    """Transcribe tests/fixtures/sample.wav and assert expected spoken keyword and metadata."""
    raw = handle_transcribe_audio({"path": str(FIXTURE_WAV)})
    result = json.loads(raw)

    assert result["status"] == "ok", f"Unexpected error: {result}"
    text_lower = result["text"].lower()
    assert any(kw in text_lower for kw in ("impressed", "tried", "super")), (
        f"Expected keyword ('impressed', 'tried', or 'super') in transcript, got: {result['text']!r}"
    )
    assert isinstance(result["duration"], (int, float))
    assert result["duration"] >= 3.0
    assert result["language"] == "en"
    assert bool(result["model"])


@pytest.mark.skipif(
    not FIXTURE_WAV.is_file() or find_whisper_python() is None or find_cached_model_path("large-v3-turbo") is None,
    reason="Skipping direct faster-whisper worker test: local whisper venv or cached model is absent.",
)
def test_integration_direct_faster_whisper_worker() -> None:
    """Verify direct faster-whisper execution when prefer_server is False."""
    set_plugin_config({
        "model_size": "large-v3-turbo",
        "compute_type": "int8",
        "language": "en",
        "prefer_server": False,
        "server_url": "",
    })
    raw = handle_transcribe_audio({"path": str(FIXTURE_WAV)})
    result = json.loads(raw)

    assert result["status"] == "ok", f"Unexpected error: {result}"
    assert result["backend"] in ("faster_whisper_venv", "faster_whisper_inproc")
    assert "impressed" in result["text"].lower() or "tried" in result["text"].lower()
    assert result["duration"] >= 3.0
    assert result["language"] == "en"
    assert result["model"] == "large-v3-turbo"


# ---------------------------------------------------------------------------
# 7. Concurrency Test (Simultaneous Transcriptions)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not FIXTURE_WAV.is_file() or not _has_real_whisper_backend(),
    reason="Skipping concurrency test: sample.wav or local Whisper model/backend is absent.",
)
def test_concurrent_transcriptions_do_not_corrupt_state(tmp_path: Path) -> None:
    """Run two simultaneous real transcriptions and verify neither corrupts the other or the cache."""
    clip_a = tmp_path / "concurrent_a.wav"
    clip_b = tmp_path / "concurrent_b.wav"
    clip_a.write_bytes(FIXTURE_WAV.read_bytes())
    clip_b.write_bytes(FIXTURE_WAV.read_bytes())

    def worker(target_path: Path) -> dict[str, Any]:
        raw = handle_transcribe_audio({"path": str(target_path)})
        return json.loads(raw)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(worker, clip_a)
        fut_b = pool.submit(worker, clip_b)
        res_a = fut_a.result(timeout=120)
        res_b = fut_b.result(timeout=120)

    for res, expected_path in ((res_a, clip_a), (res_b, clip_b)):
        assert res["status"] == "ok", f"Concurrent transcription failed: {res}"
        assert res["path"] == str(expected_path.resolve())
        assert "impressed" in res["text"].lower() or "tried" in res["text"].lower()
        assert res["duration"] >= 3.0
        assert res["language"] == "en"

    last = get_last_transcript()
    assert last is not None
    assert last["status"] == "ok"
    assert last["path"] in (str(clip_a.resolve()), str(clip_b.resolve()))
