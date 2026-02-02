bootstrap-python-env:
    uv venv
    source .venv/bin/activate.fish
    # uv pip install google-genai

generate-video:
    uv run generate_video.py

generate-audio:
    uv run generate_audio.py

generate-image:
    uv run generate_image.py

generate-speech lang='de':
    uv run generate_speech.py --lang {{lang}}

transcribe *args="--lang de --hugging-face-api-key $HUGGING_FACE_API_KEY --audio-folder ~/Drive/archive/Maxim/03-Beweismaterial/Audio":
    uv run transcribe_audio_folder_to_markdown.py {{args}}

merge-videos:
    #!/usr/bin/env bash
    set -euo pipefail
    for theme in $(ls video/video-series-*.mp4 2>/dev/null | sed 's/.*video-series-\([a-f0-9]*\)-.*/\1/' | sort -u); do
        filelist=$(mktemp)
        ls video/video-series-${theme}-*.mp4 | sort | while read -r f; do
            echo "file '$(pwd)/$f'" >> "$filelist"
        done
        output="video/video-series-${theme}-final.mp4"
        ffmpeg -y -f concat -safe 0 -i "$filelist" -c copy "$output"
        rm "$filelist"
        echo "Merged to $output"
    done

