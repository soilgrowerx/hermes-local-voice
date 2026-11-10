"""Tool schemas for hermes-local-voice."""

from __future__ import annotations

TRANSCRIBE_AUDIO_SCHEMA: dict = {
    "name": "transcribe_audio",
    "description": (
        "Transcribe a local audio file (.wav, .mp3, .ogg, .opus, .m4a, .flac, .webm) "
        "offline on this machine using faster-whisper. Returns text, duration in seconds, "
        "detected language, and the model used. Audio never leaves the machine."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Local filesystem path to the audio file to transcribe.",
            },
            "language": {
                "type": "string",
                "description": "Optional language code (for example 'en') or 'auto' for automatic detection.",
            },
            "model_size": {
                "type": "string",
                "description": "Optional Whisper model override (tiny.en, base.en, small.en, medium.en, large-v3-turbo).",
            },
            "cleanup_on_success": {
                "type": "boolean",
                "description": "If true, remove the input audio file after transcription succeeds (preserved on failure).",
            },
        },
        "required": ["path"],
    },
}

VOICE_RECORD_START_SCHEMA: dict = {
    "name": "voice_record_start",
    "description": (
        "Start local 16 kHz mono microphone recording via PipeWire (pw-record), ALSA (arecord), "
        "or ffmpeg. Degrades cleanly with a structured status when no microphone is available."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

VOICE_RECORD_STOP_SCHEMA: dict = {
    "name": "voice_record_stop",
    "description": (
        "Stop the active local microphone recording and return the saved audio file path and "
        "duration in seconds. Optionally transcribes the recording immediately."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "auto_transcribe": {
                "type": "boolean",
                "description": "If true, immediately transcribe the captured audio file upon stopping.",
            },
            "cleanup_on_transcribe_success": {
                "type": "boolean",
                "description": "If true and auto_transcribe is enabled, delete the temporary recording after transcription succeeds.",
            },
        },
    },
}

TRANSCRIBE_LAST_SCHEMA: dict = {
    "name": "transcribe_last",
    "description": (
        "Return the most recent transcript produced by hermes-local-voice, including text, "
        "duration, detected language, model, and timestamp."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}
