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
import xml.etree.ElementTree as ET
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


def extract_ssml_title(ssml: str) -> str | None:
    """Extract dc:title from SSML metadata."""
    try:
        root = ET.fromstring(ssml)
        # Search with and without namespace
        for tag in ["dc:title", "{http://purl.org/dc/elements/1.1/}title"]:
            elem = root.find(f".//{tag}")
            if elem is not None and elem.text:
                return elem.text.strip()
        # Fallback: regex for dc:title
        match = re.search(r"<dc:title>(.*?)</dc:title>", ssml)
        if match:
            return match.group(1).strip()
    except ET.ParseError:
        pass
    return None


# Prefer SSML file if it exists, otherwise fall back to markdown
file_prefix = "speech"
if SSML_FILE.exists() and SSML_FILE.read_text().strip():
    ssml_content = SSML_FILE.read_text().strip()
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_content)
    input_hash = ssml_content
    title = extract_ssml_title(ssml_content)
    if title:
        file_prefix = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "-")
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

VOICE_NAME = "de-DE-Chirp3-HD-Fenrir"

gcs_filename = f"{file_prefix}_{VOICE_NAME}_{theme}-{timestamp}.wav"
output_gcs_uri = f"gs://{GCS_BUCKET}/speech/{gcs_filename}"

client = texttospeech.TextToSpeechLongAudioSynthesizeClient()

voice = texttospeech.VoiceSelectionParams(
    name=VOICE_NAME,
    language_code="de-DE",
)

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    speaking_rate=1.0,  # 0.92? - slightly slower, professional audiobook pace
    sample_rate_hertz=48000,  # studio-quality sample rate
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
    ["ffmpeg", "-y", "-i", str(local_wav), "-codec:a", "libmp3lame", "-q:a", "0", "-ar", "48000", str(local_mp3)],
    check=True,
    capture_output=True,
)
local_wav.unlink()

print(f"Saved {local_mp3}")
