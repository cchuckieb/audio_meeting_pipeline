FROM python:3.9-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git tini \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel setuptools

# PyAV from wheel only (never compile)
RUN pip install --only-binary=:all: "av==12.3.0"

# faster-whisper WITHOUT deps (prevents it pulling av==11.*)
RUN pip install --no-deps "faster-whisper==1.1.0"

# whisperx WITHOUT deps (prevents pip pulling/trying to build av)
RUN pip install --no-deps "whisperx==3.3.1"

# now install the rest (NO whisperx / NO faster-whisper here)
RUN pip install \
    "torch==2.2.2" "torchaudio==2.2.2" \
    "pyannote.audio==3.1.1" \
    "huggingface_hub>=0.20" \
    "numpy<2" "pandas<3" "tqdm" \
    "soundfile" "ffmpeg-python" \
    "transformers<5" "sentencepiece" \
    "ctranslate2" \
    "scipy" "pydub" \
    "nltk"

WORKDIR /work
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
