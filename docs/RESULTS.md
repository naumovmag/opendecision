# Реальные результаты OpenDecision 9B

Дата эксперимента: 2026-10-12.

**QLoRA Qwen3.5-9B на RTX 4080 SUPER 16 GB выполнено успешно.**
Оба адаптера созданы реальными backward/optimizer steps и проверены после перезагрузки.

## Окружение

GPU: `NVIDIA GeForce RTX 4080 SUPER, 580.159.03, 16376 MiB, 13430 MiB`.
Python 3.12.3; PyTorch CUDA 12.8; BF16: True.

```json
{
  "torch": "2.11.0+cu128",
  "transformers": "5.17.0",
  "unsloth": "2026.10.3",
  "unsloth_zoo": "2026.10.3",
  "peft": "0.21.2",
  "trl": "1.13.0",
  "bitsandbytes": "0.50.2",
  "fastapi": "0.143.0",
  "pydantic": "2.14.1",
  "accelerate": "1.15.0",
  "triton": "3.6.0"
}
```

Официальный checkpoint: `Qwen/Qwen3.5-9B`, commit `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
NF4, double quantization, BF16 compute, rank 8, alpha 16, AdamW 8-bit.
Дополнительно по первоначальной инструкции скачана официальная 4B; в обучении и сравнении она не использовалась.

## Обучение

| Эксперимент | Steps | Train loss | Последний val loss | Peak allocated GiB | Peak reserved GiB | Время s |
|---|---:|---:|---:|---:|---:|---:|
| Smoke / 256 tokens | 30 | 0.130961 | 3.33786e-06 | 7.799 | 7.920 | 113.43 |
| Pilot / 512 tokens | 100 | 0.0571284 | 0.000215491 | 7.836 | 7.965 | 240.27 |

Обучаемых параметров: 14,548,992; доля 0.1544%.
Базовые веса и vision encoder заморожены. Проверены 30 + 100 finite/nonzero gradient checks и столько же изменений LoRA.
Максимум общей занятой памяти устройства среди NVML samples pilot: **11.771 GiB** (включая рабочий стол; это выборочный мониторинг, не непрерывный аппаратный peak).

Адаптеры: `checkpoints/qlora_smoke/adapter`, `checkpoints/qlora_pilot/adapter`.
Независимая проверка маски и loss обычным Transformers: `reports/adapter-audit.json`, PASS.
Loss имеет только два supervised tokens: метка и EOS; prefix labels = -100.
Smoke использовал 1429 train-примеров, 169 явно исключено по длине; pilot использовал 1598.

## Качество

Train/validation/test: 1598/185/217; всего 2000, splits по группам 80/10/10.
Простой синтетический test содержит числовые правила и ограниченные смысловые сценарии. Это не универсальный benchmark.

| Модель | Accuracy | Macro-F1 | NLL | Brier | ECE | P50 ms |
|---|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3.5-9B baseline | 0.8387 | 0.5802 | 0.4377 | 0.2405 | 0.0848 | 94.7078 |
| Qwen/Qwen3.5-9B + LoRA | 1.0000 | 1.0000 | 0.0045 | 0.0027 | 0.0035 | 103.4626 |
| Qwen/Qwen3.5-9B + LoRA | 1.0000 | 1.0000 | 0.0009 | 0.0001 | 0.0008 | 106.7683 |


Отдельный ручной OOD smoke-набор из 16 случаев:

| Модель | Accuracy | Macro-F1 | NLL | Brier | ECE | P50 ms |
|---|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3.5-9B baseline | 1.0000 | 1.0000 | 0.0510 | 0.0085 | 0.0482 | 71.9121 |
| Qwen/Qwen3.5-9B + LoRA | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 75.8206 |
| Qwen/Qwen3.5-9B + LoRA | 1.0000 | 1.0000 | 0.0003 | 0.0000 | 0.0003 | 78.6499 |


Одинаковые test IDs, хеш набора, prompt, NF4 и алгоритм scoring использованы до/после. На синтетическом test улучшение подтверждено; на ручном наборе accuracy уже была 100% до обучения.
Разрезы ru/en и ошибки сохранены в соответствующих JSON-отчётах. Latency включает первый cold forward, не включает загрузку весов.

## Устойчивость

| Преобразование | Baseline flip rate | Smoke flip rate |
|---|---:|---:|
| permutation | 0.0000 | 0.0000 |
| rename | 0.0000 | 0.0000 |
| paraphrase | 0.0000 | 0.0000 |
| injection | 0.1250 | 0.0625 |
| long | 0.0000 | 0.0000 |

Prompt injection остаётся реальным ограничением; системная инструкция не обеспечивает полной защиты.

## Калибровка

Temperature = 2.536649, подобрана только на 185 validation примерах; экспериментальная: True.
Порог review = 0.64, выбран на validation. Параметры: `checkpoints/calibration.json`.

| Split | NLL raw → calibrated | Brier raw → calibrated | ECE raw → calibrated |
|---|---:|---:|---:|
| validation | 0.0612627 → 0.0310131 | 0.0180359 → 0.0151318 | 0.00978595 → 0.00537086 |
| test | 0.00448638 → 0.0096841 | 0.00271591 → 0.00410233 | 0.00354538 → 0.00817156 |

На test калибровка ухудшила метрики. Поэтому работающий API использует raw probabilities и `calibrated=false`. Калиброванный режим отдельно проверен по HTTP (`reports/api-calibrated-smoke.json`), но не выбран как основной.

## Реальный HTTP API

Сервис проверен на **127.0.0.1:8001**, Swagger `/docs`. Порт 8000 занят существующим `categorytree-app-1`; этот сервис не изменялся.
`/health`, `/v1/models`, `/v1/decisions`, `/v1/decisions/batch`, `/docs` — PASS; ошибочный JSON contract — 422.

```json
{
  "model": "Qwen/Qwen3.5-9B",
  "selected": "choice_0",
  "confidence": 0.9999999994249451,
  "probabilities": {
    "choice_0": 0.9999999994249451,
    "choice_1": 4.058652124141745e-10,
    "choice_2": 1.6918979216421962e-10
  },
  "calibrated": false,
  "calibration_experimental": null,
  "requires_review": false,
  "latency_ms": 400.267162011005
}
```

## Тесты и критерии

```text
........................                                                 [100%]
=============================== warnings summary ===============================
.venv/lib/python3.12/site-packages/fastapi/testclient.py:1
  .venv/lib/python3.12/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

tests/test_gpu.py: 14 warnings
  .venv/lib/python3.12/site-packages/torch/jit/_script.py:365: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
24 passed, 15 warnings in 16.86s
```

PASS: окружение, GPU/BF16, Qwen3.5-9B NF4, динамические logits scores, 2000 примеров, групповые splits, target masks, реальные backward и optimizer steps, adapters/checkpoints, reload, baseline/after evaluation, calibration, robustness, HTTP API, ruff, compileall, pytest.
BLOCKED: запуск именно на порту 8000 — занят чужим контейнером. Функциональный HTTP тест выполнен на 8001.
Не проводилось: платное сравнение Microsoft-Decision-1, полноразмерный независимый benchmark, BF16 9B reference (ограничение VRAM).

## Ошибки, исправленные по ходу эксперимента

1. Sandbox bwrap не запускал команды; использовались разрешённые выполнения вне sandbox.
2. Медленная установка CUDA 13 заменена отдельным Python 3.12 окружением с PyTorch 2.11 CUDA 12.8 из кеша.
3. Unsloth принимал leaf names в target_modules, а не полные пути. Исправлено после проверки установленного исходного кода.
4. VLM loader возвращал Processor. Текстовый pipeline использует его tokenizer.
5. TRL default warmup_steps=0.1 давал первый шаг с lr=0. Контроль остановил такой запуск; warmup_steps=0 позволил проверить все 30 реальных обновлений.
OOM не наблюдался; подмена модели не использовалась.

Предупреждения: reference causal_conv1d/DeltaNet kernels в Transformers работают, но медленнее оптимизированных; опциональные torchao extensions не загружаются и не используются NF4 backend; Starlette/httpx и torch.jit имеют предупреждения deprecation. Конфликтов требований пакетов `uv pip check` не обнаружил.

## Воспроизведение

```bash
cd opendecision
make setup
make doctor
make dataset
make baseline
# Перед GPU командами остановить собственный OpenDecision API
.venv/bin/opendecision train --config configs/qlora_9b.yaml --resume checkpoints/qlora_smoke/checkpoint-20
make evaluate
make calibrate
OPENDECISION_GPU_TEST=1 .venv/bin/python -m pytest -q
make api PORT=8001
# В другом терминале:
.venv/bin/python scripts/http_smoke.py --port 8001
```

Для нового независимого обучения задайте новый output_dir в копии YAML. Повторный запуск без resume не перезаписывает существующий адаптер.
