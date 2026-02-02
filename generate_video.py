# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai",
# ]
# ///

import os
import time
import sys
from google import genai
from google.genai import types


client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
)

source = types.GenerateVideosSource(
    prompt="""Create a beautiful, stunning, dramatic and breath-taking intro for a cyber world war with China and Russia winning over USA and EU. """,
)

config = types.GenerateVideosConfig(
    aspect_ratio="16:9",
    number_of_videos=1,
    duration_seconds=8,
    person_generation="allow_all",
)

# Generate the video generation request
operation = client.models.generate_videos(
    model="veo-3.1-generate-preview", source=source, config=config
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
    if generated_video.video and generated_video.video.video_bytes:
        filename = f"generated_video_{i}.mp4"
        with open(filename, "wb") as f:
            f.write(generated_video.video.video_bytes)
        print(f"Saved {filename}")
