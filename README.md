# OpenDecision

A decision service built on **Qwen3.5-9B**. Input: a situation, a question and a list of options
with descriptions (2 to 26). Output: the selected option and a probability for every option.
Use cases: routing requests between assistants, product and text classification, answer
verification, prioritization, rule-based selection.

This project trained and serves the model
[**Adata-Decision-1-9B**](https://huggingface.co/NAUMOV/Adata-Decision-1-9B) (Qwen3.5-9B + LoRA).
It contains everything needed to run it as an HTTP API and to reproduce training and evaluation.

```bash
curl -s http://127.0.0.1:8000/v1/decisions -H 'Content-Type: application/json' -d '{
  "state": "Сформируй техническое задание на закупку ноутбуков",
  "question": "Какой ассистент должен обработать обращение?",
  "options": [
    {"id": "tz",    "description": "Технические спецификации закупок (ТЗ, техспеки)"},
    {"id": "court", "description": "Анализ судебных дел и документов"},
    {"id": "fines", "description": "Административные штрафы"}
  ]}'
```

```json
{"model": "Adata-Decision-1-9B", "selected": "tz", "confidence": 0.99999987,
 "probabilities": {"tz": 0.99999987, "court": 7.7e-08, "fines": 5.3e-08},
 "calibrated": false, "calibration_experimental": null, "requires_review": false,
 "latency_ms": 155.0}
```

The request above asks, in Russian, which assistant should handle "Draft a technical
specification for purchasing laptops"; the options are procurement specifications, court
cases and administrative fines. The model picks the procurement-specification assistant.
The model was trained on Russian and English.

> A functional analogue of
> [Microsoft-Decision-1](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-microsoft-decision-1-in-microsoft-foundry-for-decision-and-classific/4562742)
> (selection among dynamic options using logits). This project is **not affiliated with
> Microsoft**, does not reproduce its training and makes no claim of comparable quality.

## Contents

1. [How it works](#how-it-works)
2. [Quick start (Docker)](#quick-start-docker)
3. [API](#api)
4. [Configuration](#configuration)
5. [Performance](#performance)
6. [Quality](#quality)
7. [Running without Docker](#running-without-docker)
8. [Training and evaluation](#training-and-evaluation)
9. [Project structure](#project-structure)
10. [Limitations](#limitations)

## How it works

This is **not a chat model** and nothing is generated. Options are labeled `A`, `B`, `C`, …
and the model sees only their descriptions (the original `id`s never enter the prompt). The
decision is the logits of the labels after a fixed prompt, normalized with softmax over the
provided options only. One forward pass per request.

- `enable_thinking=False`; the system prompt and JSON layout are identical in training and inference.
- `state` is treated as untrusted data: the model ignores instructions inside it (no guarantee).
- Concurrent requests go through a **dynamic batcher**: everything that arrives within 5 ms is
  scored in one GPU pass. Requests sharing `question` and `options` compute the common part
  of the prompt once and cache it on the GPU; only `state` is re-run.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (in Russian).

## Quick start (Docker)

**Requirements:** Linux x86_64, an NVIDIA GPU with 12 GB or more (tested on RTX 5090 and
RTX 4080 SUPER), driver ≥ 570, Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/),
~20 GB of disk for weights, and `pip install -U huggingface_hub` for the `hf` command.

```bash
git clone https://github.com/naumovmag/opendecision.git && cd opendecision

# 1. The Adata-Decision-1-9B adapter (56 MB)
hf download NAUMOV/Adata-Decision-1-9B --include "adapter/*" --local-dir /tmp/adata
mkdir -p checkpoints/qlora_pilot && cp -r /tmp/adata/adapter checkpoints/qlora_pilot/

# 2. The base model (~19 GB), exactly this commit: the adapter was trained on it
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a

# 3. Build and run (the first build takes 10–20 minutes, image ~13 GB)
mkdir -p .triton-cache
export OPENDECISION_UID=$(id -u) OPENDECISION_GID=$(id -g) HF_CACHE_DIR=$HOME/.cache/huggingface
docker compose up -d --build
docker compose logs -f        # wait for "Application startup complete" (~30–60 s)

curl http://127.0.0.1:8000/health
# {"status":"ok","model":"Adata-Decision-1-9B"}
```

Swagger UI: <http://127.0.0.1:8000/docs>. Stop: `docker compose down`.

**16 GB GPU** (e.g. an RTX 4080 SUPER that also drives a desktop): use smaller batches —
`export OPENDECISION_CONFIG=configs/inference-pilot-local.yaml` before `docker compose up`.

**Network access:** `export OPENDECISION_BIND=<host IP> OPENDECISION_PORT=8001`. The API has
no authentication — expose it only on a trusted network or behind an authenticating proxy.

**A second replica on another GPU** runs as a separate compose project, see
[compose.gpu1.yaml](compose.gpu1.yaml):

```bash
OPENDECISION_GPU=1 OPENDECISION_PORT=8002 \
  docker compose -p opendecision-gpu1 -f compose.yaml -f compose.gpu1.yaml up -d --no-build
```

The container restarts with the host (`restart: unless-stopped`). After the first download
it can run offline: `export HF_HUB_OFFLINE=1`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{"status": "ok", "model": ...}` |
| GET | `/v1/models` | model, adapter, commit, package versions, batcher statistics |
| POST | `/v1/decisions` | a single decision |
| POST | `/v1/decisions/batch` | `{"requests": [...]}`, up to 16 requests |
| GET | `/docs` | Swagger UI |

**Request**

| Field | Type | Constraints |
|---|---|---|
| `state` | string | 1–24,000 chars. The situation, user message, product description |
| `question` | string | 1–2,000 chars. What to decide |
| `options` | array | 2–26 options with unique `id`s |
| `options[].id` | string | 1–128 chars, `^[\w.:-]+$`; **not** included in the prompt |
| `options[].description` | string | 1–2,000 chars. The only thing the model knows about the option |
| `confidence_threshold` | float \| null | 0–1; `requires_review` threshold for this request |

Unknown fields are rejected.

**Response**

| Field | Meaning |
|---|---|
| `selected` | `id` of the best option (always chosen, even if every option is bad) |
| `confidence` | probability of the selected option |
| `probabilities` | probabilities of all options, summing to 1 |
| `requires_review` | `confidence` is below the threshold (0.7 by default): hand the decision to a human |
| `calibrated` | whether probability calibration is applied (see [Calibration](#calibration)) |
| `calibration_experimental` | calibration was fitted on fewer than 500 examples; `null` without calibration |
| `latency_ms` | time from queueing to response |

Status codes: `200` decision made; `413` body larger than 1 MiB; `422` invalid format or prompt
longer than `max_tokens` (requests are never truncated); `503` inference error.

**Python client**

```python
import httpx

r = httpx.post("http://127.0.0.1:8000/v1/decisions", json={
    "state": "The customer wants a refund for a cancelled order",
    "question": "Which agent should handle this?",
    "options": [{"id": "refund", "description": "Refunds and cancellations"},
                {"id": "delivery", "description": "Delivery"}]})
print(r.json()["selected"])   # refund
```

More examples: [data/examples/request.json](data/examples/request.json),
[data/examples/adata_assistants.json](data/examples/adata_assistants.json) (a catalog of 6 assistants),
[scripts/client.py](scripts/client.py).

**Writing option descriptions.** The model sees only `description`, so list synonyms and
abbreviations, including lowercase spellings; avoid phrasing like "does not handle X" when X
is a frequent word in requests; add an explicit catch-all option.

## Configuration

The config is selected with `OPENDECISION_CONFIG` (a YAML file from [configs/](configs/)).

| Config | What it runs |
|---|---|
| `inference-pilot.yaml` | **Adata-Decision-1-9B**: NF4 base + adapter `checkpoints/qlora_pilot/adapter` (Docker default) |
| `inference-pilot-local.yaml` | the same for 16 GB GPUs: smaller batches |
| `inference-pilot-merged-dq.yaml` | merged BF16 model from HF (GPU ≥ 24 GB), see [without Docker](#running-without-docker) |
| `inference.yaml` | the original Qwen3.5-9B without the adapter (for comparison) |

| Setting | Default | Purpose |
|---|---|---|
| `served_model_name` | — | model name reported in API responses |
| `max_tokens` | 2048 | prompt length limit; longer requests get HTTP 422 |
| `confidence_threshold` | 0.7 | `requires_review` threshold |
| `calibration` | — | calibration file (JSON) |
| `max_batch_size` | 64 | maximum requests per GPU pass |
| `batch_wait_ms` | 5 | how long to wait for more requests after the first one |
| `max_batch_tokens` | 65536 | limit on `requests × length` per GPU pass |
| `prefix_cache_size` | 16 | shared prefixes kept on the GPU (~70 MB each) |
| `warmup` | true | warm up Triton kernels at startup (5–40 s) |

`compose.yaml` variables:

| Variable | Default | Purpose |
|---|---|---|
| `OPENDECISION_BIND` / `OPENDECISION_PORT` | `127.0.0.1` / `8000` | published API address |
| `OPENDECISION_GPU` | `0` | GPU index |
| `OPENDECISION_UID` / `OPENDECISION_GID` | `1000` | container user = owner of the HF cache and the adapter |
| `OPENDECISION_CONFIG` | `configs/inference-pilot.yaml` | model config |
| `HF_CACHE_DIR` | `~/.cache/huggingface` | host HF cache with the base model |
| `HF_HUB_OFFLINE` | `0` | `1` disables network access |

### Calibration

By default `confidence` is the model's raw confidence (`calibrated: false`): 0.95 does not
mean "right 95% of the time". Calibration (temperature scaling) fits a correction on labeled
examples so that confidence matches the actual share of correct answers, and picks a
`requires_review` threshold targeting 90% accuracy. It needs a JSONL file with correct
answers (format: [data/processed/validation.jsonl](data/processed/validation.jsonl)), ideally
300–500 real requests:

```bash
.venv/bin/python scripts/run_evaluation.py --config configs/inference-pilot.yaml \
  --dataset my-validation.jsonl --split validation --output reports/validation.json
.venv/bin/python scripts/calibrate.py --output checkpoints/calibration-pilot.json
# then in the config: calibration: checkpoints/calibration-pilot.json
```

## Performance

A single request with 2–10 options and a 200–600-token prompt:

| GPU | Config | GPU memory | Latency |
|---|---|---:|---:|
| RTX 5090 32 GB | `inference-pilot.yaml` | ~17 GB including batcher caches | 40–115 ms |
| RTX 4080 SUPER 16 GB | `inference-pilot-local.yaml` | ~11 GB | 56–116 ms |

Concurrent requests, time until the last response ([scripts/load_test.py](scripts/load_test.py)):

| Concurrent | Shared catalog (6 assistants), RTX 5090 | Unique options per request, RTX 5090 | Shared catalog, RTX 4080 SUPER |
|---:|---:|---:|---:|
| 1 | 40 ms | 115 ms | 56 ms |
| 20 | 216 ms | 2.6 s | — |
| 50 | 450–510 ms (up to ~1.1 s on a new batch shape) | 4.8–6.1 s | 1.3 s |

Requests sharing an option list (routing, classification against a catalog) are accelerated
by the prefix cache. Requests with unique options are compute-bound: ~100 ms each.
`flash-linear-attention` (in `requirements.lock`) speeds up the Gated DeltaNet layers 1.3–1.5×.

## Quality

Share of correct decisions:

| Dataset | Examples | Adata-Decision-1-9B | Original Qwen3.5-9B |
|---|---:|---:|---:|
| Synthetic test (same templates as training) | 217 | 1.000 | 0.903 |
| Domain validation: Adata assistants, products, other tasks | 135 | 0.911 | 0.800 |
| 8 catalogs absent from training | 64 | 0.969 | 0.953 |

Robustness (16 hand-written examples, share of changed decisions): option permutation 0,
`id` renaming 0, description paraphrasing 0, long irrelevant `state` 0, instruction injection
into `state` 2/16.

All datasets are small and synthetic; real user requests were not part of the evaluation.
1.000 on the synthetic test does not mean 100% on real data. Detailed reports:
[docs/RESULTS.md](docs/RESULTS.md) (in Russian), [reports/](reports/).

## Running without Docker

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
make setup      # .venv + requirements.lock (PyTorch cu128) + the opendecision package
make doctor     # checks GPU, driver, memory
# adapter and base model: steps 1–2 of the quick start
OPENDECISION_CONFIG=configs/inference-pilot.yaml .venv/bin/opendecision serve --host 127.0.0.1 --port 8000
```

The merged BF16 model (GPU ≥ 24 GB, a single directory without the adapter):

```bash
hf download NAUMOV/Adata-Decision-1-9B --exclude "adapter/*" --local-dir models/opendecision-qwen3.5-9b-pilot
OPENDECISION_CONFIG=configs/inference-pilot-merged-dq.yaml .venv/bin/opendecision serve --port 8000
```

**Never** load the merged model in 4 bits: quality drops to the level of the original model
(0.778 vs 0.911 on validation).

From Python, without HTTP:

```python
from opendecision.config import Settings
from opendecision.engine import create_engine
from opendecision.schemas import DecisionRequest

engine = create_engine(Settings.load("configs/inference-pilot.yaml"))
try:
    result = engine.decide(DecisionRequest(
        state="The customer asks for a refund", question="Which agent should handle this?",
        options=[{"id": "refund", "description": "Refunds and cancellations"},
                 {"id": "delivery", "description": "Delivery"}]))
    print(result.model_dump())
finally:
    engine.backend.close()
```

CLI: `opendecision classify data/examples/request.json --config configs/inference-pilot.yaml`,
`opendecision evaluate --config ... --dataset ...`, `opendecision train --config ...`.

## Training and evaluation

```bash
make download     # Qwen/Qwen3.5-9B
make dataset      # 2,000 deterministic synthetic examples → data/processed/
make baseline     # evaluate the original model
make train-pilot  # QLoRA, 100 steps (~4 minutes on an RTX 4080 SUPER)
make evaluate
make test         # ruff + unit tests (no GPU, no weights downloaded)
OPENDECISION_GPU_TEST=1 .venv/bin/python -m pytest tests/test_gpu.py -q   # real 9B on a GPU
```

Adata-Decision-1-9B recipe: QLoRA with an NF4 base (double quant, BF16 compute), LoRA r=8,
alpha=16, modules `q/k/v/o_proj` and `gate/up/down_proj` of the language model (vision frozen),
14.5 M trainable parameters (0.15%), 100 steps, batch 1 × accumulation 8, lr 5·10⁻⁵,
AdamW 8-bit, seed 42. The loss covers only the answer label and EOS. Config:
[configs/qlora_pilot.yaml](configs/qlora_pilot.yaml); details in
[docs/TRAINING.md](docs/TRAINING.md) (in Russian).

**Data** ([data/processed/](data/processed/)) is exactly what the model was trained on:
1,598 train / 185 validation / 217 test examples in Russian and English. Mostly numeric rules
("value 170 → range 167–176"), option permutations, instruction injections and 18 semantic
scenarios. The split is by group, so variants of one situation never land in different splits.
[data/examples/independent.jsonl](data/examples/independent.jsonl) holds 16 hand-written
examples outside training.

To fine-tune on your own data, use the same JSONL format (`Sample` in
[src/opendecision/training/dataset.py](src/opendecision/training/dataset.py)) and set the paths
under `datasets:` in the training config. Real labeled requests help far more than synthetic data.

## Project structure

```text
src/opendecision/
  api.py                  FastAPI: /health, /v1/models, /v1/decisions
  engine.py               dynamic batcher, probabilities, requires_review
  prompting.py            prompt and A–Z labels
  backends/               Qwen3.5 loading (NF4/BF16 + LoRA), scoring, prefix cache
  training/               dataset, synthetic data generation, QLoRA training
  evaluation/             metrics, comparison, robustness
  calibration.py          temperature scaling
configs/                  inference and training configs
scripts/                  evaluation, calibration, load test, adapter merging
data/                     the model's dataset and request examples
tests/                    unit and GPU tests
reports/                  evaluation and training reports
Dockerfile, compose.yaml  API container
```

Code comments, server error messages and `docs/` are in Russian.

## Limitations

- **Lowercase abbreviations.** Russian «тз» (lowercase "technical specification") tokenizes
  differently from «ТЗ» and may be routed to the wrong assistant with low confidence
  (`requires_review=true`). List spellings in option descriptions.
- **Prompt injection.** Instructions in `state` are treated as data, but protection is incomplete.
- **Text only.** The vision part of the model was not trained or evaluated.
- **Long requests are rejected** (422), not truncated.
- **Without calibration** `confidence` is not the probability of being correct.

## License

[Apache-2.0](LICENSE), same as the base model [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).
