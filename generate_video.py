# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai",
# ]
# ///

import glob
import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types

VIDEO_DIR = Path("video")
VIDEO_DIR.mkdir(exist_ok=True)

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
)

prompt = """
  Create a documentary about the migration to Titan and
  elaborate why it could be a better option than a migration to Mars.
"""

theme = hashlib.sha256(prompt.encode()).hexdigest()[:8]

# Find the latest video in the video/ subfolder to continue from
previous_videos = sorted(glob.glob(str(VIDEO_DIR / f"video-series-{theme}-*.mp4")))
if previous_videos:
    latest_video_path = previous_videos[-1]
    print(f"Continuing from: {latest_video_path}")
    with open(latest_video_path, "rb") as f:
        previous_bytes = f.read()
    source = types.GenerateVideosSource(
        video=types.Video(video_bytes=previous_bytes, mime_type="video/mp4"),
        prompt=prompt,
    )
else:
    print("No previous video found, generating from prompt only.")
    source = types.GenerateVideosSource(prompt=prompt)

config = types.GenerateVideosConfig(
    aspect_ratio="16:9",
    number_of_videos=1,
    durationSeconds=8,
    personGeneration="allow_adult",
    # generate_audio=True, # not available in veo-3.1-fast-generate-preview
    resolution="720p",
    # seed=0,
)

# Generate the video generation request
operation = client.models.generate_videos(
    model="veo-3.1-fast-generate-preview", source=source, config=config
)

# Waiting for the video(s) to be generated
while not operation.done:
    print("Video has not been generated yet. Check again in 10 seconds...")
    time.sleep(10)
    operation = client.operations.get(operation)

response = operation.result
if not response:
    print("Error occurred while generating video.")
    sys.exit(1)

generated_videos = response.generated_videos
if not generated_videos:
    print("No videos were generated.")
    sys.exit(1)

print(f"Generated {len(generated_videos)} video(s).")
for i, generated_video in enumerate(generated_videos):
    video = generated_video.video
    if not video:
        print(f"Video {i}: no video object")
        continue
    # Download the video to populate video_bytes
    client.files.download(file=video)
    if video.video_bytes:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        filename = VIDEO_DIR / f"video-series-{theme}-{timestamp}.mp4"
        with open(filename, "wb") as f:
            f.write(video.video_bytes)
        print(f"Saved {filename}")
    else:
        print(f"Video {i}: no video bytes available (uri: {video.uri})")
