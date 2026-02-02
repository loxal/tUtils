bootstrap-python-env:
    uv venv
    source .venv/bin/activate.fish

generate-video:
    uv run generate_video.py
