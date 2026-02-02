# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-cloud-texttospeech",
# ]
# ///

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import texttospeech

SPEECH_DIR = Path("speech")
SPEECH_DIR.mkdir(exist_ok=True)

PROMPT_FILE = Path.home() / "my/src/loxal/lox/al/prompts/sppech.md"
prompt = PROMPT_FILE.read_text().strip()

if not prompt:
    print(f"Prompt file is empty: {PROMPT_FILE}")
    sys.exit(1)

theme = hashlib.sha256(prompt.encode()).hexdigest()[:8]

client = texttospeech.TextToSpeechClient()

voice = texttospeech.VoiceSelectionParams(
    name="en-US-Chirp3-HD-Achird",
    language_code="en-US",
)

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3,
)

print("Generating speech...")
response = client.synthesize_speech(
    input=texttospeech.SynthesisInput(text=prompt),
    voice=voice,
    audio_config=audio_config,
)

if not response.audio_content:
    print("No audio was generated.")
    sys.exit(1)

timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
filename = SPEECH_DIR / f"speech-series-{theme}-{timestamp}.mp3"
with open(filename, "wb") as f:
    f.write(response.audio_content)
print(f"Saved {filename}")
