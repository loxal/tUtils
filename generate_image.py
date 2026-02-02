# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai",
# ]
# ///

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types

IMAGE_DIR = Path("image")
IMAGE_DIR.mkdir(exist_ok=True)

client = genai.Client(
    vertexai=True,
    project="instant-droplet-485818-i0",
    location="us-central1",
)

PROMPT_FILE = Path.home() / "my/src/loxal/lox/al/prompts/avatar.md"
prompt = PROMPT_FILE.read_text().strip()

theme = hashlib.sha256(prompt.encode()).hexdigest()[:8]

config = types.GenerateImagesConfig(
    numberOfImages=4,
    aspectRatio="1:1",
    negativePrompt="blurry, low quality, distorted",
    personGeneration="allow_all",
    addWatermark=True,
    outputMimeType="image/png",
)

print("Generating image(s)...")
response = client.models.generate_images(
    model="imagen-4.0-ultra-generate-001",
    prompt=prompt.strip(),
    config=config,
)

if not response.generated_images:
    print("No images were generated.")
    sys.exit(1)

print(f"Generated {len(response.generated_images)} image(s).")
for i, generated_image in enumerate(response.generated_images):
    image = generated_image.image
    if not image:
        print(f"Image {i}: no image object")
        continue
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    filename = IMAGE_DIR / f"image-series-{theme}-{timestamp}-{i}.png"
    if image.image_bytes:
        with open(filename, "wb") as f:
            f.write(image.image_bytes)
        print(f"Saved {filename}")
    elif image.gcs_uri:
        import subprocess
        subprocess.run(
            ["gcloud", "storage", "cp", image.gcs_uri, str(filename)],
            check=True,
        )
        print(f"Saved {filename} (from {image.gcs_uri})")
    else:
        print(f"Image {i}: no image bytes or URI available")
