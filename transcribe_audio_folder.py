# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai-whisper",
#     "tqdm",
#     "pyannote.audio",
#     "torch",
# ]
# ///

"""
Audio-to-SSML Transcription Script
===================================
Uses OpenAI Whisper large-v3 with optional speaker diarization.

Features:
- Auto-downloads model to ~/Downloads/whisper_models/
- Recursively processes all audio files in a folder
- Creates .ssml files alongside source audio files (compatible with TTS engines)
- Speaker diarization (requires HuggingFace token for pyannote)
- **Parallel processing**: diarization and transcription run concurrently
- Progress display

Requirements:
    brew install ffmpeg          # macOS - REQUIRED for audio decoding

For speaker diarization (optional):
    Requires HuggingFace token with pyannote access

Supported formats: m4a, aac, mp3, wav, flac, ogg, wma, opus, webm, mp4
"""

import os
import sys
import shutil
import warnings
import argparse
from pathlib import Path
from datetime import timedelta
from xml.sax.saxutils import escape
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    setup_model_cache()
    
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

        # PyTorch 2.6+ defaults weights_only=True which breaks pyannote checkpoints
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

        # Use 'token=' (new API) instead of deprecated 'use_auth_token='
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        pipeline.to(torch.device(device))
        torch.load = _orig_load  # restore original
        print(f"✓ Speaker diarization ready (using {device})")
        return pipeline

    except ImportError:
        print("\n⚠ pyannote.audio not installed. Speaker diarization disabled.")
        print("  To enable: uv pip install pyannote.audio torch")
        return None
    except Exception as e:
        print(f"\n⚠ Could not load diarization: {e}")
        return None


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    import subprocess
    import json
    
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    return None


def load_audio_with_ffmpeg(audio_path: str, target_sr: int = 16000):
    """Load any audio format using ffmpeg, return torch tensor and sample rate."""
    import subprocess
    import numpy as np
    import torch
    
    # Use ffmpeg to decode to raw PCM (mono, 16kHz, 16-bit signed)
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-f", "s16le",        # 16-bit signed little-endian PCM
        "-acodec", "pcm_s16le",
        "-ar", str(target_sr), # Resample to target
        "-ac", "1",            # Mono
        "-v", "quiet",
        "-"                    # Output to stdout
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    
    # Convert bytes to numpy array, then to torch tensor
    audio_data = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(audio_data).unsqueeze(0)  # Add channel dimension
    
    return waveform, target_sr


def run_diarization(audio_path: str, diarization_pipeline) -> dict:
    """Run speaker diarization on audio file. Returns speaker segments dict."""
    if diarization_pipeline is None:
        return {}

    try:
        # Get actual audio duration to prevent boundary errors
        duration = get_audio_duration(audio_path)
        
        # Load audio with ffmpeg (handles all formats including m4a, aac)
        waveform, sample_rate = load_audio_with_ffmpeg(audio_path, target_sr=16000)
        
        if duration:
            # Trim 0.5s from end to avoid boundary errors
            trim_samples = int(0.5 * sample_rate)
            if waveform.shape[1] > trim_samples:
                waveform = waveform[:, :-trim_samples]
        
        diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
        
        segments = {}
        # Handle various pyannote API versions
        if hasattr(diarization, 'itertracks'):
            # Standard pyannote.core.Annotation API
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = min(turn.start, turn.end)
                segments[(start, turn.end)] = speaker
        elif hasattr(diarization, 'speaker_diarization') and diarization.speaker_diarization is not None:
            # DiarizeOutput dataclass (pyannote 3.x) - speaker_diarization is an Annotation
            for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                start = min(turn.start, turn.end)
                segments[(start, turn.end)] = speaker
        elif hasattr(diarization, 'annotation'):
            # DiarizeOutput wrapper - access the underlying annotation
            for turn, _, speaker in diarization.annotation.itertracks(yield_label=True):
                start = min(turn.start, turn.end)
                segments[(start, turn.end)] = speaker
        elif hasattr(diarization, 'segments'):
            # Another possible API
            for seg in diarization.segments:
                start = min(seg.start, seg.end)
                segments[(start, seg.end)] = seg.speaker
        else:
            # Debug: print what we got
            print(f"  ⚠ Unknown diarization output type: {type(diarization)}, attrs: {dir(diarization)}")
        return segments
    except Exception as e:
        print(f"  ⚠ Diarization failed: {e}")
        return {}


def run_transcription(audio_path: str, model, language: str = None) -> dict:
    """Run Whisper transcription. Returns dict with 'segments' and 'language'."""
    options = {"verbose": False}
    if language:
        options["language"] = language

    return model.transcribe(audio_path, **options)


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


def voices_for_language(lang_code: str) -> list[str]:
    """Return 6 distinct Chirp 3 HD GA voices for a language."""
    # GA voices available across all 31 Chirp 3 HD locales
    voice_names = ["Fenrir", "Aoede", "Orus", "Puck", "Charon", "Kore"]
    locale_map = {
        "de": "de-DE", "en": "en-US", "es": "es-ES",
        "fr": "fr-FR", "ru": "ru-RU",
    }
    locale = locale_map.get(lang_code, f"{lang_code}-{lang_code.upper()}")
    return [f"{locale}-Chirp3-HD-{name}" for name in voice_names]


def transcribe_audio(
    audio_path: str,
    model,
    diarization_pipeline=None,
    language: str = None
) -> str:
    """
    Transcribe audio file to SSML format with parallel diarization.

    Returns an SSML string with <voice> tags for speakers and <break> for pauses,
    compatible with Google Cloud Text-to-Speech and generate_speech.py.
    """
    # Run diarization and transcription in parallel
    speaker_segments = {}
    result = None
    
    if diarization_pipeline is not None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            diarization_future = executor.submit(
                run_diarization, audio_path, diarization_pipeline
            )
            transcription_future = executor.submit(
                run_transcription, audio_path, model, language
            )
            
            # Collect results
            speaker_segments = diarization_future.result()
            result = transcription_future.result()
    else:
        # No diarization, just transcribe
        result = run_transcription(audio_path, model, language)
    
    speaker_labels = build_speaker_labels(speaker_segments) if speaker_segments else {}

    detected_lang = result.get("language", "en")
    voice_pool = voices_for_language(detected_lang)
    title = Path(audio_path).stem

    # Assign a distinct voice to each speaker (up to 6)
    speaker_voice_map = {}
    for i, label in enumerate(speaker_labels.values()):
        speaker_voice_map[label] = voice_pool[i % len(voice_pool)]

    # Build SSML
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"')
    lines.append('       xmlns:dc="http://purl.org/dc/elements/1.1/"')
    lines.append(f'       xml:lang="{detected_lang}">')
    lines.append(f"  <dc:title>{escape(title)}</dc:title>")
    lines.append("")

    # Document the speaker-to-voice mapping
    if speaker_voice_map:
        for label, voice_name in speaker_voice_map.items():
            lines.append(f"  <!-- {label}: {voice_name} -->")
        lines.append("")

    current_speaker = None
    voice_open = False
    prev_end = 0.0

    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()

        if not text:
            continue

        # Determine speaker
        raw_speaker = find_speaker_for_segment(start, end, speaker_segments)
        speaker = speaker_labels.get(raw_speaker) if raw_speaker else None

        # Insert a break for pauses > 0.5s between segments
        gap = start - prev_end
        if prev_end > 0 and gap > 0.5:
            pause_ms = min(int(gap * 1000), 5000)  # cap at 5s
            indent = "    " if voice_open else "  "
            lines.append(f'{indent}<break time="{pause_ms}ms"/>')

        # Switch voice on speaker change
        if speaker_labels and speaker != current_speaker:
            if voice_open:
                lines.append("  </voice>")
                lines.append("")
            voice_name = speaker_voice_map.get(speaker, voice_pool[0])
            lines.append(f'  <!-- {speaker} -->')
            lines.append(f'  <voice name="{voice_name}">')
            current_speaker = speaker
            voice_open = True

        indent = "    " if voice_open else "  "
        lines.append(f"{indent}{escape(text)}")
        prev_end = end

    if voice_open:
        lines.append("  </voice>")

    lines.append("</speak>")
    return "\n".join(lines)

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
    if diarization_pipeline is not None:
        print("  ℹ Diarization + transcription will run in parallel\n")

    # Process each file with progress bar
    successful = 0
    failed = 0

    for audio_path in tqdm(audio_files, desc="Transcribing", unit="file"):
        try:
            # Create output path (same name, .ssml extension)
            ssml_path = audio_path.with_suffix(".ssml")

            # Skip if already transcribed
            if ssml_path.exists():
                tqdm.write(f"  ⏭ Skipping (already exists): {ssml_path.name}")
                continue

            tqdm.write(f"  → Processing: {audio_path.name}")

            # Transcribe
            ssml = transcribe_audio(
                str(audio_path),
                model,
                diarization_pipeline,
                language
            )

            # Write to temp file first, then atomic rename to prevent half-written files
            temp_path = ssml_path.with_suffix(".ssml.tmp")
            temp_path.write_text(ssml, encoding="utf-8")
            temp_path.rename(ssml_path)  # Atomic on POSIX
            tqdm.write(f"  ✓ Created: {ssml_path.name}")
            successful += 1

        except Exception as e:
            tqdm.write(f"  ✗ Failed: {audio_path.name} - {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"COMPLETE: {successful} transcribed, {failed} failed")
    print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(description="Audio-to-SSML Transcription")
    parser.add_argument("folder", nargs="?", help="Folder containing audio files")
    parser.add_argument("--audio-folder", help="Folder containing audio files")
    parser.add_argument("--lang", help="Language code (e.g., 'en', 'de')")
    parser.add_argument("--hugging-face-api-key", help="HuggingFace API Key for diarization")
    args = parser.parse_args()

    if args.hugging_face_api_key:
        os.environ["HF_TOKEN"] = args.hugging_face_api_key

    print("="*60)
    print("   AUDIO TO SSML TRANSCRIPTION")
    print("   Using OpenAI Whisper large-v3")
    print("="*60)

    # Check ffmpeg first
    check_ffmpeg()

    # Load Whisper model
    model = load_whisper_model("large-v3")

    # Try to set up diarization
    diarization = setup_diarization()

    # Folder selection
    folder = None
    folder_input = args.audio_folder or args.folder
    if folder_input:
        folder_arg = Path(folder_input).expanduser().resolve()
        if folder_arg.is_dir():
            folder = folder_arg
            print(f"\nProcessing folder: {folder}")
        else:
            print(f"\n✗ Error: Not a valid directory: {folder_arg}")
            sys.exit(1)

    if not folder:
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
    language = args.lang
    if language:
        print(f"\nLanguage set to: {language}")
    elif args.audio_folder or args.folder:
        # Non-interactive mode if folder provided via CLI
        print("\nLanguage: Auto-detect")
    else:
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
