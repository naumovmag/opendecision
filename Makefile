PY := .venv/bin/python
PORT ?= 8000
.PHONY: help setup doctor download dataset baseline train-smoke train-pilot evaluate calibrate test api docker-build docker-up docker-down docker-logs
help:
	@echo 'setup doctor download dataset baseline train-smoke train-pilot evaluate calibrate test api docker-build docker-up docker-down docker-logs'
setup:
	test -x .venv/bin/python || uv venv .venv --python /usr/bin/python3
	uv pip install --python $(PY) -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
	uv pip install --python $(PY) --no-deps -e .
doctor:
	$(PY) scripts/doctor.py
download:
	$(PY) scripts/download_models.py --model Qwen/Qwen3.5-9B
dataset:
	$(PY) scripts/generate_dataset.py
	$(PY) scripts/create_examples.py
baseline:
	$(PY) scripts/run_evaluation.py
	$(PY) scripts/run_evaluation.py --dataset data/processed/test.jsonl --output reports/baseline-test.json
train-smoke:
	$(PY) scripts/train_smoke.py
train-pilot:
	$(PY) scripts/train_smoke.py --config configs/qlora_pilot.yaml
evaluate:
	$(PY) scripts/run_evaluation.py --config configs/inference-adapter.yaml --output reports/fine-tuned.json
calibrate:
	$(PY) scripts/run_evaluation.py --config configs/inference-adapter.yaml --dataset data/processed/validation.jsonl --split validation --output reports/validation.json
	$(PY) scripts/calibrate.py --output checkpoints/calibration.json
test:
	.venv/bin/ruff check src scripts tests
	$(PY) -m compileall -q src scripts tests
	$(PY) -m pytest -q
api:
	OPENDECISION_CONFIG=configs/inference-adapter.yaml .venv/bin/opendecision serve --host 127.0.0.1 --port $(PORT)
docker-build:
	docker compose build
docker-up:
	docker compose up -d
docker-down:
	docker compose down
docker-logs:
	docker compose logs -f opendecision
