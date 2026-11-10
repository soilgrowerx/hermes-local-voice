# hermes-local-voice

Offline voice capture and transcription plugin for Hermes Agent (`>=0.21.5`) powered by `faster-whisper` and `PipeWire` (`pw-record`), `ALSA` (`arecord`), or `ffmpeg`.

## Disclosure

Audio files and microphone recordings processed by `hermes-local-voice` are transcribed locally on your machine and are not uploaded to external servers. No API keys, cloud tokens, or remote endpoints are used during transcription. The only network activity occurs when you explicitly run `hermes voice --download <model>` to fetch a model from Hugging Face into `HERMES_HOME`.

## What It Does

`hermes-local-voice` adds four agent tools, two inbound chat hooks, an in-session `/local-voice` slash command, and a headless `hermes voice` CLI subcommand:

* **Tools:**
  * `transcribe_audio(path)`: validates and transcribes a local audio file (`.wav`, `.mp3`, `.ogg`, `.opus`, `.m4a`, `.flac`, `.webm`, `.aac`, `.mp4`, `.wma`, `.caf`). Returns transcript text, duration in seconds, detected language, and the model name.
  * `voice_record_start()`: starts 16 kHz mono microphone capture via `pw-record`, `arecord`, or `ffmpeg`. Degrades cleanly with a structured status message on headless machines without a microphone.
  * `voice_record_stop(auto_transcribe=False, cleanup_on_transcribe_success=False)`: stops active microphone capture and returns the saved audio path and duration in seconds.
  * `transcribe_last()`: returns the most recent transcript from memory or `HERMES_HOME/plugin-data/hermes-local-voice/last_transcript.json`.
* **Message Hooks:**
  * `pre_gateway_dispatch`: automatically transcribes inbound voice notes and audio attachments arriving through messaging gateways into text before agent dispatch.
  * `pre_llm_call`: automatically transcribes local audio file attachments referenced in CLI or desktop chat turns and injects the transcript context.
* **Headless CLI:**
  * `hermes voice <file>` prints the transcript to standard output without starting an interactive chat session.

## Why Local Processing Matters

Cloud speech-to-text APIs require network access, account credentials, and per-minute billing, and they transmit raw voice audio to third-party servers. Running `faster-whisper` with CTranslate2 `int8` quantization on the local CPU or loopback worker keeps voice notes inside the local environment, works on air-gapped or offline machines, and requires zero API keys.

## Dependencies and Supported Platforms

| Name | Purpose | Requirement |
| :--- | :--- | :--- |
| Python `>=3.11` | Plugin runtime | Required |
| `faster-whisper` (`>=1.0.0,<2.0.0`) | Local Whisper inference engine | Required (in active Python env, `$HERMES_HOME/venvs/whisper`, `~/.venvs/whisper`, or loopback server on `127.0.0.1:8001`) |
| `ffmpeg` / `ffprobe` | Format conversion and container probing for `.mp3`, `.ogg`, `.m4a`, `.flac`, `.webm` | Recommended (required for non-WAV formats) |
| `pw-record` or `arecord` | 16 kHz mono microphone recording | Optional (required only for live mic capture) |

### Supported Platforms

* **Desktop Hosts (`x86_64`, `arm64`):** full support for file transcription and live microphone capture via PipeWire (`pw-record`), ALSA (`arecord`), or PulseAudio (`ffmpeg`).
* **macOS (`arm64`, `x86_64`):** full support for file transcription and microphone capture via `ffmpeg` (`avfoundation`).
* **Headless Hosts:** file transcription and inbound chat hooks work without audio hardware; `voice_record_start` reports `status: "unavailable"` with an actionable message when no audio capture device is present.

## Install

Install from Git or a local checkout using the Hermes plugin manager:

```bash
hermes plugins install https://github.com/georgessteward/hermes-local-voice --yes-deps --enable
```

For a local repository checkout:

```bash
hermes plugins install file:///path/to/hermes-local-voice --yes-deps --enable
```

## First-Run Model Download and Disk Cost

`hermes-local-voice` is offline by default and does not download model weights unannounced during transcription. Models are stored under `$HERMES_HOME/plugin-data/hermes-local-voice/models` (or read from an existing local Hugging Face cache at `~/.cache/huggingface/hub`).

| Model Name | Approximate Disk Size | Notes |
| :--- | :--- | :--- |
| `tiny.en` / `tiny` | 75 MB | small CPU footprint for quick voice notes |
| `base.en` / `base` | 145 MB | default CPU model balancing speed and accuracy |
| `small.en` / `small` | 466 MB | balanced accuracy for technical vocabulary |
| `medium.en` / `medium` | 1500 MB | larger model, slower on CPU |
| `large-v3-turbo` | 1600 MB | pruned large-v3 decoder suitable for local server or GPU |
| `large-v3` | 3100 MB | full large-v3 weights |

Check cached models and disk cost before downloading:

```bash
hermes voice --models
```

Download a model explicitly (announces disk cost before fetching and can be cancelled with `Ctrl+C`):

```bash
hermes voice --download base.en
```

## CLI Usage

Transcribe an audio file headlessly:

```bash
hermes voice sample.wav
```

Emit structured JSON with duration, detected language, and model used:

```bash
hermes voice sample.wav --json
```

Retrieve the most recent transcript:

```bash
hermes voice --last
```

Record from the microphone and transcribe:

```bash
hermes voice --record-start
hermes voice --record-stop
```

## Configuration

Configure settings in `~/.hermes/config.yaml` under `plugins.entries.hermes-local-voice.config`:

```yaml
plugins:
  entries:
    hermes-local-voice:
      enabled: true
      config:
        model_size: "base.en"
        compute_type: "int8"
        language: "en"
        model_dir: ""
        max_file_mb: 100
        server_url: "http://127.0.0.1:8001"
        prefer_server: true
        auto_transcribe_inbound: true
```

## Temporary File Policy

When converting non-WAV audio files or transcribing temporary recordings:

* Temporary converted audio files are removed automatically after transcription succeeds.
* If transcription fails, temporary audio files are preserved on disk and their path is returned in the `preserved_audio_path` field so no recording is lost.

## License

MIT
