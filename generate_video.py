# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai",
# ]
# ///

import argparse
import glob
import hashlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types

parser = argparse.ArgumentParser()
parser.add_argument("--project", default="instant-droplet-485818-i0")
parser.add_argument("--gcs-bucket", default=None,
                    help="GCS bucket for output staging (e.g. gs://my-bucket). "
                         "Required by Vertex AI for large video output.")
parser.add_argument("--input", dest="input_mode", default="text",
                    choices=["text", "jpg", "png", "video"],
                    help="Input mode: text (prompt only), jpg/png (image-to-video), or video (video extension)")
parser.add_argument("--prompt-file", required=True,
                    help="Path to a text/markdown file containing the prompt")
parser.add_argument("--resolution", default="720p", choices=["720p", "1080p"],
                    help="Output video resolution (default: 720p)")
parser.add_argument("--model", default="veo-3.1-generate-001",
                    help="Video generation model (default: veo-3.1-generate-001)")
args = parser.parse_args()

VIDEO_DIR = Path("video")
VIDEO_DIR.mkdir(exist_ok=True)

client = genai.Client(
    vertexai=True,
    project=args.project,
    location="us-central1",
)

prompt = Path(args.prompt_file).read_text().strip()

theme = hashlib.sha256(prompt.encode()).hexdigest()[:8]

gcs_output_uri = f"{args.gcs_bucket}/video-staging/" if args.gcs_bucket else None

if args.input_mode == "video":
    previous_videos = sorted(glob.glob(str(VIDEO_DIR / f"video-series-{theme}-*.mp4")))
    if not previous_videos:
        print("No previous video found for extension. Run with --input text first.")
        sys.exit(1)
    latest_video_path = previous_videos[-1]
    print(f"Extending from video: {latest_video_path}")
    continuation_prompt = f"Continue this video the best possible way for the initial prompt: {prompt.strip()}"
    config = types.GenerateVideosConfig(
        number_of_videos=1,
        person_generation="allow_all",
        generate_audio=True,
        resolution=args.resolution,
        output_gcs_uri=gcs_output_uri,
    )
    operation = client.models.generate_videos(
        model=args.model,
        video=types.Video(
            video_bytes=Path(latest_video_path).read_bytes(),
            mime_type="video/mp4",
        ),
        prompt=continuation_prompt,
        config=config,
    )
elif args.input_mode in ("jpg", "png"):
    ext = args.input_mode
    mime_type = "image/jpeg" if ext == "jpg" else "image/png"
    images = sorted(glob.glob(str(VIDEO_DIR / f"*.{ext}")))
    if not images:
        print(f"No .{ext} files found in video/ folder.")
        sys.exit(1)
    latest_image_path = images[-1]
    print(f"Generating video from image: {latest_image_path}")
    config = types.GenerateVideosConfig(
        aspect_ratio="16:9",
        number_of_videos=1,
        duration_seconds=8,
        person_generation="allow_all",
        generate_audio=True,
        resolution=args.resolution,
    )
    operation = client.models.generate_videos(
        model=args.model,
        image=types.Image(
            image_bytes=Path(latest_image_path).read_bytes(),
            mime_type=mime_type,
        ),
        prompt=prompt,
        config=config,
    )
else:
    print("Generating from text prompt only.")
    config = types.GenerateVideosConfig(
        aspect_ratio="16:9",
        number_of_videos=1,
        duration_seconds=8,
        person_generation="allow_all",
        generate_audio=True,
        resolution=args.resolution,
    )
    operation = client.models.generate_videos(
        model=args.model,
        prompt=prompt,
        config=config,
    )

# Waiting for the video(s) to be generated
while not operation.done:
    print("Video has not been generated yet. Check again in 10 seconds...")
    time.sleep(10)
    operation = client.operations.get(operation)

if operation.error:
    print(f"Video generation failed: {operation.error}")
    sys.exit(1)

response = operation.result
if not response:
    print(f"Error occurred while generating video. Operation: {operation}")
    sys.exit(1)

generated_videos = response.generated_videos
if not generated_videos:
    print(f"No videos were generated. Response: {response}")
    sys.exit(1)

print(f"Generated {len(generated_videos)} video(s).")
for i, generated_video in enumerate(generated_videos):
    video = generated_video.video
    if not video:
        print(f"Video {i}: no video object")
        continue
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    filename = VIDEO_DIR / f"video-series-{theme}-{timestamp}.mp4"
    if video.video_bytes:
        with open(filename, "wb") as f:
            f.write(video.video_bytes)
        print(f"Saved {filename}")
    elif video.uri:
        # Vertex AI returns a GCS URI; download via gcloud
        subprocess.run(
            ["gcloud", "storage", "cp", video.uri, str(filename)],
            check=True,
        )
        print(f"Saved {filename} (from {video.uri})")
    else:
        print(f"Video {i}: no video bytes or URI available")
