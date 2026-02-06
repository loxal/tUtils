# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-cloud-texttospeech",
#     "markdown-it-py",
# ]
# ///

import argparse
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

LANG_VOICES = {
    "de": {"voice": "de-DE-Neural2-B", "language_code": "de-DE"},
    "en": {"voice": "en-US-Neural2-D", "language_code": "en-US"},
    "ru": {"voice": "ru-RU-Neural2-B", "language_code": "ru-RU"},
}

parser = argparse.ArgumentParser()
parser.add_argument("--lang", default="de", choices=LANG_VOICES.keys())
parser.add_argument("--author", help="Author/Artist name for metadata")
parser.add_argument("--override-voice", help="Override the default voice name for the selected language")
parser.add_argument("--strip-pitch", action="store_true", help="Remove pitch attributes from prosody tags (required for Studio voices)")
parser.add_argument("--strip-emphasis", action="store_true", help="Remove emphasis tags (required for Studio voices)")
args = parser.parse_args()

lang_cfg = LANG_VOICES[args.lang]
VOICE_NAME = lang_cfg["voice"]
LANGUAGE_CODE = lang_cfg["language_code"]

if args.override_voice:
    VOICE_NAME = args.override_voice
    print(f"Voice overridden to: {VOICE_NAME}")


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


def strip_pitch_from_prosody(ssml: str) -> str:
    """Remove pitch attributes from prosody tags (unsupported by Studio voices)."""
    return re.sub(r'(<prosody[^>]*)\s+pitch="[^"]*"', r'\1', ssml)


def strip_emphasis_tags(ssml: str) -> str:
    """Remove emphasis tags but keep their content (unsupported by Studio voices)."""
    return re.sub(r'<emphasis[^>]*>(.*?)</emphasis>', r'\1', ssml, flags=re.DOTALL)


def sanitize_ssml_for_google(ssml: str) -> str:
    """Strip W3C SSML features unsupported by Google Cloud TTS.

    Removes: XML declaration, DOCTYPE, namespace attributes, <metadata> block,
    <voice> wrapper. Keeps: <speak>, <p>, <s>, <break>, <emphasis>, <prosody>.
    """
    # Remove XML declaration and DOCTYPE
    s = re.sub(r"<\?xml[^?]*\?>", "", ssml)
    s = re.sub(r"<!DOCTYPE[^>]*>", "", s)
    # Remove <metadata>...</metadata> block (with any attributes)
    s = re.sub(r"<metadata[^>]*>.*?</metadata>", "", s, flags=re.DOTALL)
    # Unwrap <voice ...>...</voice> (keep inner content)
    s = re.sub(r"<voice[^>]*>", "", s)
    s = re.sub(r"</voice>", "", s)
    # Simplify <speak> tag — remove all attributes except xml:lang
    lang_match = re.search(r'xml:lang="([^"]*)"', s)
    lang_attr = f' xml:lang="{lang_match.group(1)}"' if lang_match else ""
    s = re.sub(r"<speak[^>]*>", f"<speak{lang_attr}>", s)
    return s.strip()


def extract_dc_metadata(ssml: str) -> dict[str, str]:
    """Extract all Dublin Core metadata elements from SSML.

    Supported DC elements: title, creator, subject, description,
    publisher, contributor, date, type, format, identifier,
    source, language, relation, coverage, rights.
    """
    dc_fields = [
        "title", "creator", "subject", "description", "publisher",
        "contributor", "date", "type", "format", "identifier",
        "source", "language", "relation", "coverage", "rights",
    ]
    metadata = {}
    # Regex approach — works regardless of XML parser namespace handling
    for field in dc_fields:
        match = re.search(rf"<dc:{field}>(.*?)</dc:{field}>", ssml, re.DOTALL)
        if match:
            metadata[field] = match.group(1).strip()
    return metadata


# Prefer SSML file if it exists, otherwise fall back to markdown
file_prefix = "speech"
dc_metadata = {}
if SSML_FILE.exists() and SSML_FILE.read_text().strip():
    ssml_content = SSML_FILE.read_text().strip()
    google_ssml = sanitize_ssml_for_google(ssml_content)
    if args.strip_pitch:
        google_ssml = strip_pitch_from_prosody(google_ssml)
        print("Pitch attributes stripped from prosody tags")
    if args.strip_emphasis:
        google_ssml = strip_emphasis_tags(google_ssml)
        print("Emphasis tags stripped")
    synthesis_input = texttospeech.SynthesisInput(ssml=google_ssml)
    input_hash = ssml_content
    dc_metadata = extract_dc_metadata(ssml_content)
    if dc_metadata.get("title"):
        file_prefix = re.sub(r"[^\w\s-]", "", dc_metadata["title"]).strip().replace(" ", "-")
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

gcs_filename = f"{file_prefix}_{VOICE_NAME}_{theme}-{timestamp}.wav"
output_gcs_uri = f"gs://{GCS_BUCKET}/speech/{gcs_filename}"

client = texttospeech.TextToSpeechLongAudioSynthesizeClient(transport="rest")

voice = texttospeech.VoiceSelectionParams(
    name=VOICE_NAME,
    language_code=LANGUAGE_CODE,
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

# Convert WAV to MP3 with embedded metadata
local_mp3 = local_wav.with_suffix(".mp3")
ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(local_wav), "-codec:a", "libmp3lame", "-q:a", "0", "-ar", "48000"]

if args.author:
    dc_metadata["creator"] = args.author

# Map Dublin Core metadata to ID3 tags
dc_to_id3 = {
    "title": "title",
    "creator": "artist",
    "subject": "genre",
    "description": "comment",
    "publisher": "publisher",
    "contributor": "album_artist",
    "date": "date",
    "language": "language",
    "rights": "copyright",
    "source": "url",
}
for dc_field, id3_tag in dc_to_id3.items():
    value = dc_metadata.get(dc_field)
    if value:
        ffmpeg_cmd.extend(["-metadata", f"{id3_tag}={value}"])

ffmpeg_cmd.append(str(local_mp3))
subprocess.run(
    ffmpeg_cmd,
    check=True,
    capture_output=True,
)
local_wav.unlink()

print(f"Saved {local_mp3}")
