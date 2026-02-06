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

# Alternative voice: de-DE-Chirp3-HD-Fenrir
generate-speech lang='de' override-voice='true' strip-pitch='false' strip-emphasis='false':
    uv run generate_speech.py --lang {{lang}} {{ if override-voice == "true" { "--override-voice de-DE-Neural2-B" } else { "" } }} {{ if strip-pitch == "true" { "--strip-pitch" } else { "" } }} {{ if strip-emphasis == "true" { "--strip-emphasis" } else { "" } }}

transcribe *args="--lang de --hugging-face-api-key $HUGGING_FACE_API_KEY --audio-folder ~/Drive/archive/Maxim/03-Beweismaterial/Audio":
    uv run transcribe_audio_folder.py {{args}}
    # uv run --python 3.12 transcribe_audio_folder.py {{args}}

gooogle-auth:
    gcloud auth login
    gcloud auth application-default login
    gcloud auth application-default set-quota-project instant-droplet-485818-i0

    gcloud services enable aiplatform.googleapis.com --project=instant-droplet-485818-i0
    gcloud services enable texttospeech.googleapis.com --project=instant-droplet-485818-i0

    # gcloud projects add-iam-policy-binding instant-droplet-485818-i0 \
    #   --member="user:alexander.orlov@loxal.net" \
    #   --role="roles/aiplatform.user"

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

