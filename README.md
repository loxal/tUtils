# tUtils - Terminal Utilities

A collection of standalone scripts and utilities.

## Contents

### `transcribe_audio_folder_to_markdown.py`

Batch audio-to-Markdown transcription tool using OpenAI Whisper large-v3. Recursively processes all audio files in a folder and creates `.md` files alongside each source file with timestamped transcripts.

**Features:**

- Supports 12+ audio formats (m4a, mp3, wav, flac, ogg, opus, and more)
- Auto-downloads the Whisper large-v3 model (~3GB) to `~/Downloads/whisper_models/`
- Automatic language detection
- Optional speaker diarization via pyannote.audio (requires a HuggingFace token)
- Skips already-transcribed files on re-runs
- Progress bar via tqdm

### `hurl-to-har-to-hurl-converter/`

Bidirectional converter between [HURL](https://hurl.dev) and [HAR](http://www.softwareishard.com/blog/har-12-spec/) file formats. Auto-detects conversion direction from the input file extension. Built with Rust using `hurl_core`.

- HURL to HAR: parses HURL files and outputs HAR v1.2 JSON
- HAR to HURL: supports HAR v1.2/v1.3, filters browser-internal headers, maps cookies and body

See [`hurl-to-har-to-hurl-converter/README.md`](hurl-to-har-to-hurl-converter/README.md) for build and usage instructions.

### Google Cloud Media Generation Scripts

`generate_video.py`, `generate_audio.py`, `generate_image.py`, and `generate_speech.py` use Google Cloud Vertex AI APIs. They require Google Cloud authentication and a quota project.

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

Replace `YOUR_PROJECT_ID` and `YOUR_EMAIL` with your actual values.

#### Running

```sh
just generate-video
just generate-audio
just generate-image
just generate-speech
```

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- FFmpeg (required for audio decoding)

Install FFmpeg:

```sh
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### Installing dependencies

Install the required packages with `uv`:

```sh
uv pip install openai-whisper tqdm
```

For optional speaker diarization support:

```sh
uv pip install pyannote.audio torch
```

### Running

```sh
python transcribe_audio_folder_to_markdown.py
```

The script will interactively prompt for the folder to process and an optional language hint.
