#!/usr/bin/env python3
"""
Audio-to-Markdown Transcription Script
======================================
Uses OpenAI Whisper large-v3 with optional speaker diarization.

Features:
- Auto-downloads model to ~/Downloads/whisper_models/
- Recursively processes all audio files in a folder
- Creates .md files alongside source audio files
- Speaker diarization (requires HuggingFace token for pyannote)
- Progress display

Requirements:
    brew install ffmpeg          # macOS - REQUIRED for audio decoding
    uv pip install openai-whisper tqdm

For speaker diarization (optional):
    uv pip install pyannote.audio torch

Supported formats: m4a, aac, mp3, wav, flac, ogg, wma, opus, webm, mp4
"""

import os
import sys
import shutil
import warnings
from pathlib import Path
from datetime import timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Audio extensions to process
AUDIO_EXTENSIONS = {
    '.m4a', '.aac', '.mp3', '.wav', '.flac', '.ogg',
    '.wma', '.opus', '.webm', '.mp4', '.mpeg', '.mpga'
}

def check_ffmpeg():
    """Check if ffmpeg is installed."""
    if shutil.which("ffmpeg") is None:
        print("\n" + "="*60)
        print("ERROR: ffmpeg not found!")
        print("="*60)
        print("Whisper requires ffmpeg for audio decoding.")
        print()
        print("Install on macOS:  brew install ffmpeg")
        print("Install on Ubuntu: sudo apt install ffmpeg")
        print("="*60)
        sys.exit(1)
    print("✓ ffmpeg found")

def setup_model_cache():
    """Set up model cache directory in ~/Downloads/whisper_models/"""
    cache_dir = Path.home() / "Downloads" / "whisper_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    print(f"Model cache directory: {cache_dir}")
    return cache_dir

def load_whisper_model(model_name: str = "large-v3"):
    """Load Whisper model with progress indication."""
    import whisper

    print(f"\nLoading Whisper {model_name} model...")
    print("(This will download ~3GB on first run)")

    model = whisper.load_model(model_name)
    print(f"✓ Model '{model_name}' loaded successfully!\n")
    return model

def setup_diarization():
    """
    Set up speaker diarization pipeline.
    Returns pipeline or None if not available.
    """
    try:
        from pyannote.audio import Pipeline
        import torch

        # Check for HuggingFace token
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        if not hf_token:
            print("\n" + "="*60)
            print("SPEAKER DIARIZATION SETUP")
            print("="*60)
            print("To enable speaker identification, you need a HuggingFace token.")
            print("1. Create account at https://huggingface.co")
            print("2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("3. Get token from https://huggingface.co/settings/tokens")
            print()
            hf_token = input("Enter HuggingFace token (or press Enter to skip): ").strip()

            if not hf_token:
                print("→ Skipping speaker diarization")
                return None

        print("Loading speaker diarization model...")
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # Use 'token=' (new API) instead of deprecated 'use_auth_token='
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        pipeline.to(torch.device(device))
        print(f"✓ Speaker diarization ready (using {device})")
        return pipeline

    except ImportError:
        print("\n⚠ pyannote.audio not installed. Speaker diarization disabled.")
        print("  To enable: uv pip install pyannote.audio torch")
        return None
    except Exception as e:
        print(f"\n⚠ Could not load diarization: {e}")
        return None

def get_speaker_segments(audio_path: str, diarization_pipeline) -> dict:
    """Get speaker segments from audio file."""
    if diarization_pipeline is None:
        return {}

    try:
        diarization = diarization_pipeline(audio_path)
        segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments[(turn.start, turn.end)] = speaker
        return segments
    except Exception as e:
        print(f"  ⚠ Diarization failed: {e}")
        return {}

def find_speaker_for_segment(start: float, end: float, speaker_segments: dict) -> str:
    """Find the speaker for a given time segment."""
    if not speaker_segments:
        return None

    best_overlap = 0
    best_speaker = None

    for (seg_start, seg_end), speaker in speaker_segments.items():
        # Calculate overlap
        overlap_start = max(start, seg_start)
        overlap_end = min(end, seg_end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker

    return best_speaker

def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(int(td.total_seconds()), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def build_speaker_labels(speaker_segments: dict) -> dict:
    """Map raw speaker IDs to friendly labels (Speaker A, Speaker B, ...)."""
    raw_ids = sorted(set(speaker_segments.values()))
    labels = {}
    for i, raw_id in enumerate(raw_ids):
        letter = chr(ord("A") + i) if i < 26 else str(i + 1)
        labels[raw_id] = f"Speaker {letter}"
    return labels


def transcribe_audio(
    audio_path: str,
    model,
    diarization_pipeline=None,
    language: str = None
) -> str:
    """
    Transcribe audio file to markdown format.

    Returns markdown string with timestamps and optional speaker labels.
    """
    # Get speaker segments if diarization is available
    speaker_segments = get_speaker_segments(audio_path, diarization_pipeline)
    speaker_labels = build_speaker_labels(speaker_segments) if speaker_segments else {}

    # Transcribe with Whisper
    options = {"verbose": False}
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)

    # Build markdown content
    md_lines = []
    md_lines.append(f"# Transcript: {Path(audio_path).name}\n")
    md_lines.append(f"**Language detected:** {result.get('language', 'unknown')}\n")

    if speaker_labels:
        md_lines.append(f"**Speakers identified:** {len(speaker_labels)}\n")
        for raw_id, label in speaker_labels.items():
            md_lines.append(f"- **{label}** ({raw_id})")
        md_lines.append("")

    md_lines.append("---\n")

    current_speaker = None

    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()

        if not text:
            continue

        timestamp = format_timestamp(start)

        # Get speaker if diarization is available
        raw_speaker = find_speaker_for_segment(start, end, speaker_segments)
        speaker = speaker_labels.get(raw_speaker) if raw_speaker else None

        if speaker and speaker != current_speaker:
            current_speaker = speaker
            md_lines.append(f"\n### {speaker}\n")

        md_lines.append(f"**[{timestamp}]** {text}\n")

    return "\n".join(md_lines)

def find_audio_files(folder: Path) -> list:
    """Recursively find all audio files in folder."""
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(folder.rglob(f"*{ext}"))
        audio_files.extend(folder.rglob(f"*{ext.upper()}"))
    return sorted(set(audio_files))

def process_folder(
    folder: Path,
    model,
    diarization_pipeline=None,
    language: str = None
):
    """Process all audio files in folder and subfolders."""
    from tqdm import tqdm

    audio_files = find_audio_files(folder)

    if not audio_files:
        print(f"No audio files found in {folder}")
        return

    print(f"\nFound {len(audio_files)} audio file(s) to process:\n")
    for f in audio_files[:10]:
        print(f"  • {f.relative_to(folder)}")
    if len(audio_files) > 10:
        print(f"  ... and {len(audio_files) - 10} more")

    print()

    # Process each file with progress bar
    successful = 0
    failed = 0

    for audio_path in tqdm(audio_files, desc="Transcribing", unit="file"):
        try:
            # Create output path (same name, .md extension)
            md_path = audio_path.with_suffix(".md")

            # Skip if already transcribed
            if md_path.exists():
                tqdm.write(f"  ⏭ Skipping (already exists): {md_path.name}")
                continue

            tqdm.write(f"  → Processing: {audio_path.name}")

            # Transcribe
            markdown = transcribe_audio(
                str(audio_path),
                model,
                diarization_pipeline,
                language
            )

            # Write markdown file
            md_path.write_text(markdown, encoding="utf-8")
            tqdm.write(f"  ✓ Created: {md_path.name}")
            successful += 1

        except Exception as e:
            tqdm.write(f"  ✗ Failed: {audio_path.name} - {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"COMPLETE: {successful} transcribed, {failed} failed")
    print(f"{'='*50}")

def main():
    print("="*60)
    print("   AUDIO TO MARKDOWN TRANSCRIPTION")
    print("   Using OpenAI Whisper large-v3")
    print("="*60)

    # Check ffmpeg first
    check_ffmpeg()

    # Set up model cache
    setup_model_cache()

    # Load Whisper model
    model = load_whisper_model("large-v3")

    # Try to set up diarization
    diarization = setup_diarization()

    # Interactive folder selection
    print("\n" + "-"*60)
    while True:
        folder_path = input("\nEnter full path to audio folder: ").strip()

        # Handle quotes and expand ~
        folder_path = folder_path.strip("'\"")
        folder = Path(folder_path).expanduser().resolve()

        if folder.is_dir():
            break
        else:
            print(f"✗ Not a valid directory: {folder}")
            print("  Please enter a valid folder path.")

    # Optional language hint
    print("\nLanguage options:")
    print("  - Press Enter for auto-detection")
    print("  - Or enter language code (e.g., 'en', 'de', 'es', 'fr')")
    language = input("Language [auto]: ").strip() or None

    # Process the folder
    process_folder(folder, model, diarization, language)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
