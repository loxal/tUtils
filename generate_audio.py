# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-auth",
#     "requests",
# ]
# ///

import argparse
import base64
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import google.auth
import google.auth.transport.requests
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--prompt-file", required=True,
                    help="Path to a text/markdown file containing the prompt")
parser.add_argument("--project", default="instant-droplet-485818-i0")
parser.add_argument("--sample-count", type=int, default=1,
                    help="Number of audio clips to generate (default: 1)")
args = parser.parse_args()

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

LOCATION = "us-central1"
MODEL = "lyria-002"

raw = Path(args.prompt_file).read_text()
metadata = {}
if "---" in raw:
    meta_section, prompt = raw.split("---", 1)
    for line in meta_section.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    prompt = prompt.strip()
else:
    prompt = raw.strip()

theme = hashlib.sha256(prompt.encode()).hexdigest()[:8]

# Authenticate using application default credentials
creds, _ = google.auth.default()
auth_req = google.auth.transport.requests.Request()
creds.refresh(auth_req)

endpoint = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
    f"projects/{args.project}/locations/{LOCATION}/"
    f"publishers/google/models/{MODEL}:predict"
)

request_body = {
    "instances": [
        {
            "prompt": prompt.strip(),
            "negative_prompt": "vocals, singing, voice",
            # "seed": 42,  # for reproducible output; cannot use with sample_count
            "sample_count": args.sample_count,  # number of clips; cannot use with seed
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
