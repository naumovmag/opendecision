FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/opendecision/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0

# gcc и libc6-dev нужны Triton: ядра flash-linear-attention при первом запуске собирают CUDA-заглушку.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 opendecision \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin opendecision

WORKDIR /app
COPY requirements.lock pyproject.toml ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
       -r requirements.lock

COPY configs ./configs
COPY src ./src
COPY scripts ./scripts
RUN python -m pip install --no-deps .

USER opendecision
EXPOSE 8000
CMD ["opendecision", "serve", "--host", "0.0.0.0", "--port", "8000"]
