# tUtils - Terminal Utilities

A collection of standalone scripts and utilities. All Python scripts use PEP 723 inline metadata — `uv run` installs dependencies automatically.

## Contents

### `transcribe_audio_folder_to_markdown.py`

Batch audio-to-SSML transcription using OpenAI Whisper large-v3. Recursively processes audio files in a folder and creates `.ssml` files alongside each source, compatible with `generate_speech.py` for re-synthesis.

- Supports 12+ audio formats (m4a, mp3, wav, flac, ogg, opus, etc.)
- Auto-downloads the Whisper large-v3 model (~3GB) to `~/Downloads/whisper_models/`
- Speaker diarization via pyannote.audio — assigns distinct Chirp 3 HD voices per speaker (up to 6)
- Dublin Core metadata (`<dc:title>`) embedded in SSML output
- Skips already-transcribed files on re-runs

Diarization requires a HuggingFace token (`HF_TOKEN` env var) and accepted access on:
- `huggingface.co/pyannote/speaker-diarization-3.1`
- `huggingface.co/pyannote/segmentation-3.0`
- `huggingface.co/pyannote/speaker-diarization-community-1`

### Google Cloud Media Generation Scripts

| Script | Model | API |
|---|---|---|
| `generate_video.py` | Veo 3.1 (`veo-3.1-generate-001`) | Vertex AI |
| `generate_audio.py` | Lyria 002 (`lyria-002`) | REST |
| `generate_image.py` | Imagen 4.0 Ultra (`imagen-4.0-ultra-generate-001`) | google-genai SDK |
| `generate_speech.py` | Chirp 3 HD | Cloud Text-to-Speech Long Audio Synthesis |

**`generate_video.py`** — Generates videos with image-to-video continuation (extracts last frame via ffmpeg for chaining). Outputs to `video/` with `video-series-{theme}-{timestamp}.mp4` naming.

**`generate_audio.py`** — Music generation with prompt, negative prompt, seed, and sample count parameters. Outputs to `audio/`.

**`generate_image.py`** — Reads prompt from `~/my/src/loxal/lox/al/prompts/avatar.md`. Outputs to `image/`.

**`generate_speech.py`** — Reads SSML from `~/my/src/loxal/lox/al/prompts/speech.ssml` (falls back to `speech.md`). Supports `--lang` flag for language/voice selection (de, en, ru). Extracts Dublin Core metadata from SSML and embeds it as ID3 tags in the output MP3. Uses GCS bucket for long audio staging, converts LINEAR16 WAV to MP3 via ffmpeg.

#### Google Cloud Authentication

1. Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install).

2. Authenticate and set the quota project:

```sh
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

3. Enable the required APIs:

```sh
gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT_ID
gcloud services enable texttospeech.googleapis.com --project=YOUR_PROJECT_ID
```

4. Ensure your account has the `Vertex AI User` role:

```sh
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:YOUR_EMAIL" \
  --role="roles/aiplatform.user"
```

### `hurl-to-har-to-hurl-converter/`

Bidirectional converter between [HURL](https://hurl.dev) and [HAR](http://www.softwareishard.com/blog/har-12-spec/) formats. See [`hurl-to-har-to-hurl-converter/README.md`](hurl-to-har-to-hurl-converter/README.md).

## Setup

- Python 3.10+, [uv](https://docs.astral.sh/uv/), FFmpeg

```sh
brew install ffmpeg   # macOS
```

## Running

All recipes are in the `justfile`:

```sh
just generate-video
just generate-audio
just generate-image
just generate-speech           # default: German
just generate-speech en        # English
just generate-speech ru        # Russian
just transcribe
just merge-videos
```
