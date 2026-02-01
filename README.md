# Snippets

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
