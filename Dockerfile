# Application image only. Postgres is external (DATABASE_URL). The default build
# installs just requirements.txt: the cloud ASR/LLM/TTS backends need no native
# libraries and no torch, so the image stays small.
#
# To bake in on-device ASR (funasr / whisper.cpp, ~1 GB more), build with
#   docker build --build-arg LOCAL_ASR=1 .
# and bind-mount the weights at /app/model (see docker-compose.yml).

FROM python:3.12-slim

ARG LOCAL_ASR=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so source edits don't invalidate the install layer.
COPY requirements.txt requirements-local-asr.txt ./
RUN if [ "$LOCAL_ASR" = "1" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 libportaudio2 \
        && rm -rf /var/lib/apt/lists/* \
        # CPU wheels first; the default index serves the CUDA build, several GB the VM can't use.
        && pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
        && pip install -r requirements-local-asr.txt; \
    else \
        pip install -r requirements.txt; \
    fi

COPY . .

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/logs /app/model \
    && chown -R app:app /app
USER app

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8081/', timeout=4)"

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8081"]
