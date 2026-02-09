bootstrap-python-env:
    uv venv
    source .venv/bin/activate.fish
    # uv pip install google-genai

crate-bucket:
    gcloud storage buckets create gs://instant-droplet-485818-i0-video-staging --project=instant-droplet-485818-i0 --location=us-central1

generate-video input='text' prompt-file='/Users/alex/my/src/loxal/lox/al/prompts/video.md' resolution='720p' model='veo-3.1-generate-001' project='instant-droplet-485818-i0' gcs-bucket='gs://instant-droplet-485818-i0-video-staging':
    uv run generate_video.py --input {{input}} --prompt-file {{prompt-file}} --resolution {{resolution}} --model {{model}} --project {{project}} --gcs-bucket {{gcs-bucket}}

generate-audio:
    uv run generate_audio.py

generate-image:
    uv run generate_image.py

# Might depend on `gooogle-auth` recipe; Alternative voice: de-DE-Neural2-B (full SSML support)
generate-speech lang='de' override-voice='false' strip-pitch='false' strip-emphasis='false':
    uv run generate_speech.py --lang {{lang}} {{ if override-voice == "true" { "--override-voice de-DE-Chirp3-HD-Fenrir" } else { "" } }} {{ if strip-pitch == "true" { "--strip-pitch" } else { "" } }} {{ if strip-emphasis == "true" { "--strip-emphasis" } else { "" } }}

transcribe *args="--lang de --hugging-face-api-key $HUGGING_FACE_API_KEY --audio-folder ~/Drive/archive/Maxim/03-Beweismaterial/Audio":
    uv run transcribe_audio_folder.py {{args}}
    # uv run --python 3.12 transcribe_audio_folder.py {{args}}

gooogle-auth project='instant-droplet-485818-i0':
    gcloud auth login
    gcloud auth application-default login
    gcloud auth application-default set-quota-project {{project}}

    gcloud services enable aiplatform.googleapis.com --project={{project}}
    gcloud services enable texttospeech.googleapis.com --project={{project}}

    # gcloud projects add-iam-policy-binding {{project}} \
    #   --member="user:alexander.orlov@loxal.net" \
    #   --role="roles/aiplatform.user"

loop-video loops='10':
    #!/usr/bin/env bash
    set -euo pipefail
    input=$(ls video/*.mp4 2>/dev/null | sort | head -1)
    if [ -z "$input" ]; then
        echo "No .mp4 files found in video/ folder."
        exit 1
    fi
    echo "Looping $input {{loops}} times..."
    filelist=$(mktemp)
    for i in $(seq 1 {{loops}}); do
        echo "file '$(pwd)/$input'" >> "$filelist"
    done
    output="video/loop-{{loops}}-$(basename "$input")"
    ffmpeg -y -f concat -safe 0 -i "$filelist" -c copy "$output"
    rm "$filelist"
    echo "Saved $output"

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

