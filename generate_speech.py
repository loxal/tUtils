# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-cloud-texttospeech",
#     "markdown-it-py",
# ]
# ///

import hashlib
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import texttospeech
from markdown_it import MarkdownIt

SPEECH_DIR = Path("speech")
SPEECH_DIR.mkdir(exist_ok=True)

PROJECT_ID = "instant-droplet-485818-i0"
LOCATION = "us-central1"
GCS_BUCKET = "video-1312312uuio323"

PROMPTS_DIR = Path.home() / "my/src/loxal/lox/al/prompts"
SSML_FILE = PROMPTS_DIR / "speech.ssml"
MD_FILE = PROMPTS_DIR / "speech.md"


def markdown_to_plain_text(md: str) -> str:
    """Convert markdown to plain text by extracting text tokens."""
    md_parser = MarkdownIt()
    tokens = md_parser.parse(md)
    parts = []
    for token in tokens:
        if token.children:
            for child in token.children:
                if child.type in ("text", "softbreak", "hardbreak"):
                    parts.append(child.content if child.type == "text" else "\n")
                elif child.type == "code_inline":
                    parts.append(child.content)
        elif token.type == "fence" or token.type == "code_block":
            continue  # skip code blocks entirely
    text = "".join(parts)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Prefer SSML file if it exists, otherwise fall back to markdown
if SSML_FILE.exists() and SSML_FILE.read_text().strip():
    ssml_content = SSML_FILE.read_text().strip()
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_content)
    input_hash = ssml_content
    print(f"Using SSML input: {SSML_FILE}")
elif MD_FILE.exists() and MD_FILE.read_text().strip():
    prompt = markdown_to_plain_text(MD_FILE.read_text().strip())
    if not prompt:
        print(f"Prompt file produced no text: {MD_FILE}")
        sys.exit(1)
    synthesis_input = texttospeech.SynthesisInput(text=prompt)
    input_hash = prompt
    print(f"Using markdown input: {MD_FILE}")
else:
    print(f"No input found. Provide either {SSML_FILE} or {MD_FILE}")
    sys.exit(1)

theme = hashlib.sha256(input_hash.encode()).hexdigest()[:8]
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
gcs_filename = f"speech-series-{theme}-{timestamp}.wav"
output_gcs_uri = f"gs://{GCS_BUCKET}/speech/{gcs_filename}"

client = texttospeech.TextToSpeechLongAudioSynthesizeClient()

voice = texttospeech.VoiceSelectionParams(
    name="de-DE-Chirp3-HD-Orus",
    language_code="de-DE",
)

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    speaking_rate=1.0,
)

request = texttospeech.SynthesizeLongAudioRequest(
    parent=f"projects/{PROJECT_ID}/locations/{LOCATION}",
    input=synthesis_input,
    audio_config=audio_config,
    voice=voice,
    output_gcs_uri=output_gcs_uri,
)

print(f"Generating speech ({len(input_hash)} chars)...")
operation = client.synthesize_long_audio(request=request)

print("Waiting for long audio synthesis to complete...")
operation.result(timeout=600)

# Download WAV from GCS
local_wav = SPEECH_DIR / gcs_filename
subprocess.run(
    ["gcloud", "storage", "cp", output_gcs_uri, str(local_wav)],
    check=True,
)

# Clean up GCS file
subprocess.run(
    ["gcloud", "storage", "rm", output_gcs_uri],
    check=True,
    capture_output=True,
)

# Convert WAV to MP3
local_mp3 = local_wav.with_suffix(".mp3")
subprocess.run(
    ["ffmpeg", "-y", "-i", str(local_wav), "-codec:a", "libmp3lame", "-q:a", "2", str(local_mp3)],
    check=True,
    capture_output=True,
)
local_wav.unlink()

print(f"Saved {local_mp3}")
