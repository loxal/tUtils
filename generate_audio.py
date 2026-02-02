# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-auth",
#     "requests",
# ]
# ///

import base64
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import google.auth
import google.auth.transport.requests
import requests

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

PROJECT_ID = "instant-droplet-485818-i0"
LOCATION = "us-central1"
MODEL = "lyria-002"

prompt = """
  Write a psychedelic melodic track for a video of a cozy dystopian room,
  up high in a skyscraper where a lonely cyberpunk is writing code to free himself.
"""

theme = hashlib.sha256(prompt.encode()).hexdigest()[:8]

# Authenticate using application default credentials
creds, _ = google.auth.default()
auth_req = google.auth.transport.requests.Request()
creds.refresh(auth_req)

endpoint = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
    f"projects/{PROJECT_ID}/locations/{LOCATION}/"
    f"publishers/google/models/{MODEL}:predict"
)

request_body = {
    "instances": [
        {
            "prompt": prompt.strip(),
            "negative_prompt": "vocals, singing, voice",
            # "seed": 42,  # for reproducible output; cannot use with sample_count
            "sample_count": 4,  # number of clips; cannot use with seed
        }
    ],
    "parameters": {},
}

print("Generating audio...")
response = requests.post(
    endpoint,
    headers={
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
    },
    json=request_body,
)

if not response.ok:
    print(f"Error {response.status_code}: {response.text}")
    sys.exit(1)

predictions = response.json().get("predictions", [])
if not predictions:
    print("No audio was generated.")
    sys.exit(1)

print(f"Generated {len(predictions)} audio clip(s).")
for i, pred in enumerate(predictions):
    audio_b64 = pred.get("bytesBase64Encoded")
    if not audio_b64:
        print(f"Clip {i}: no audio data")
        continue
    audio_bytes = base64.b64decode(audio_b64)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    filename = AUDIO_DIR / f"audio-series-{theme}-{timestamp}.wav"
    with open(filename, "wb") as f:
        f.write(audio_bytes)
    print(f"Saved {filename}")
