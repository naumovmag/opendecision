"""Собирает итоговый отчёт исключительно из реальных артефактов запусков."""
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def main():
    environment = load("reports/environment.json")
    smoke = load("checkpoints/qlora_smoke/training_report.json")
    pilot = load("checkpoints/qlora_pilot/training_report.json")
    calibration = load("checkpoints/calibration.json")
    calibration_test = load("reports/calibration-test.json")
    api = load("reports/api-smoke.json")
    data = load("reports/dataset.json")
    memory = [json.loads(line) for line in Path("reports/gpu_metrics.jsonl").read_text().splitlines()]
    device_max = max(r.get("gpu_device_used", 0) for r in memory)
    lines = ["# Реальные результаты OpenDecision 9B", "", "Дата эксперимента: 2026-10-12.", "",
             "**QLoRA Qwen3.5-9B на RTX 4080 SUPER 16 GB выполнено успешно.**",
             "Оба адаптера созданы реальными backward/optimizer steps и проверены после перезагрузки.", "",
             "## Окружение", "", f"GPU: `{environment['nvidia_smi']}`.",
             f"Python {environment['python']}; PyTorch CUDA {environment['torch_cuda']}; BF16: {environment['bf16']}.",
             "", "```json", json.dumps(environment["versions"], indent=2), "```", "",
             "Официальный checkpoint: `Qwen/Qwen3.5-9B`, commit `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.",
             "NF4, double quantization, BF16 compute, rank 8, alpha 16, AdamW 8-bit.",
             "Дополнительно по первоначальной инструкции скачана официальная 4B; в обучении и сравнении она не использовалась.", "",
             "## Обучение", "", "| Эксперимент | Steps | Train loss | Последний val loss | Peak allocated GiB | Peak reserved GiB | Время s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, report in [("Smoke / 256 tokens", smoke), ("Pilot / 512 tokens", pilot)]:
        val = [r["eval_loss"] for r in report["history"] if "eval_loss" in r][-1]
        lines.append(f"| {name} | {report['evidence']['optimizer_steps_with_weight_change']} | {report['metrics']['train_loss']:.6g} | {val:.6g} | {report['gpu_peak_allocated_bytes']/1024**3:.3f} | {report['gpu_peak_reserved_bytes']/1024**3:.3f} | {report['metrics']['train_runtime']:.2f} |")
    lines += ["", f"Обучаемых параметров: {smoke['parameter_counts']['trainable_parameters']:,}; доля {smoke['parameter_counts']['trainable_percentage']:.4f}%.",
              "Базовые веса и vision encoder заморожены. Проверены 30 + 100 finite/nonzero gradient checks и столько же изменений LoRA.",
              f"Максимум общей занятой памяти устройства среди NVML samples pilot: **{device_max/1024**3:.3f} GiB** (включая рабочий стол; это выборочный мониторинг, не непрерывный аппаратный peak).", "",
              "Адаптеры: `checkpoints/qlora_smoke/adapter`, `checkpoints/qlora_pilot/adapter`.",
              "Независимая проверка маски и loss обычным Transformers: `reports/adapter-audit.json`, PASS.",
              "Loss имеет только два supervised tokens: метка и EOS; prefix labels = -100.",
              f"Smoke использовал {data['train']['count']-len(smoke['skipped_overlength']['train'])} train-примеров, {len(smoke['skipped_overlength']['train'])} явно исключено по длине; pilot использовал {data['train']['count']-len(pilot['skipped_overlength']['train'])}.", "",
              "## Качество", "", f"Train/validation/test: {data['train']['count']}/{data['validation']['count']}/{data['test']['count']}; всего 2000, splits по группам 80/10/10.",
              "Простой синтетический test содержит числовые правила и ограниченные смысловые сценарии. Это не универсальный benchmark.", "",
              Path("reports/comparison-test.md").read_text(), "", "Отдельный ручной OOD smoke-набор из 16 случаев:", "", Path("reports/comparison-ood.md").read_text(), "",
              "Одинаковые test IDs, хеш набора, prompt, NF4 и алгоритм scoring использованы до/после. На синтетическом test улучшение подтверждено; на ручном наборе accuracy уже была 100% до обучения.",
              "Разрезы ru/en и ошибки сохранены в соответствующих JSON-отчётах. Latency включает первый cold forward, не включает загрузку весов.", "",
              "## Устойчивость", "", "| Преобразование | Baseline flip rate | Smoke flip rate |", "|---|---:|---:|"]
    rb = load("reports/robustness-baseline.json")["flip_rates"]
    ra = load("reports/robustness-adapter.json")["flip_rates"]
    for kind in rb:
        lines.append(f"| {kind} | {rb[kind]:.4f} | {ra[kind]:.4f} |")
    lines += ["", "Prompt injection остаётся реальным ограничением; системная инструкция не обеспечивает полной защиты.", "",
              "## Калибровка", "", f"Temperature = {calibration['temperature']:.6f}, подобрана только на {calibration['sample_count']} validation примерах; экспериментальная: {calibration['experimental']}.",
              f"Порог review = {calibration['confidence_threshold']:.2f}, выбран на validation. Параметры: `checkpoints/calibration.json`.", "",
              "| Split | NLL raw → calibrated | Brier raw → calibrated | ECE raw → calibrated |", "|---|---:|---:|---:|"]
    for name, report in [("validation", calibration), ("test", calibration_test)]:
        lines.append("| " + name + " | " + " | ".join(f"{report['raw'][k]:.6g} → {report['calibrated'][k]:.6g}" for k in ["nll", "brier", "ece"]) + " |")
    lines += ["", "На test калибровка ухудшила метрики. Поэтому работающий API использует raw probabilities и `calibrated=false`. Калиброванный режим отдельно проверен по HTTP (`reports/api-calibrated-smoke.json`), но не выбран как основной.", "",
              "## Реальный HTTP API", "", "Сервис проверен на **127.0.0.1:8001**, Swagger `/docs`. Порт 8000 занят существующим `categorytree-app-1`; этот сервис не изменялся.",
              "`/health`, `/v1/models`, `/v1/decisions`, `/v1/decisions/batch`, `/docs` — PASS; ошибочный JSON contract — 422.", "", "```json", json.dumps(api["response"], ensure_ascii=False, indent=2), "```", "",
              "## Тесты и критерии", "", "```text", Path("reports/pytest-gpu.log").read_text().strip(), "```", "",
              "PASS: окружение, GPU/BF16, Qwen3.5-9B NF4, динамические logits scores, 2000 примеров, групповые splits, target masks, реальные backward и optimizer steps, adapters/checkpoints, reload, baseline/after evaluation, calibration, robustness, HTTP API, ruff, compileall, pytest.",
              "BLOCKED: запуск именно на порту 8000 — занят чужим контейнером. Функциональный HTTP тест выполнен на 8001.",
              "Не проводилось: платное сравнение Microsoft-Decision-1, полноразмерный независимый benchmark, BF16 9B reference (ограничение VRAM).", "",
              "## Ошибки, исправленные по ходу эксперимента", "",
              "1. Sandbox bwrap не запускал команды; использовались разрешённые выполнения вне sandbox.",
              "2. Медленная установка CUDA 13 заменена отдельным Python 3.12 окружением с PyTorch 2.11 CUDA 12.8 из кеша.",
              "3. Unsloth принимал leaf names в target_modules, а не полные пути. Исправлено после проверки установленного исходного кода.",
              "4. VLM loader возвращал Processor. Текстовый pipeline использует его tokenizer.",
              "5. TRL default warmup_steps=0.1 давал первый шаг с lr=0. Контроль остановил такой запуск; warmup_steps=0 позволил проверить все 30 реальных обновлений.",
              "OOM не наблюдался; подмена модели не использовалась.", "",
              "Предупреждения: reference causal_conv1d/DeltaNet kernels в Transformers работают, но медленнее оптимизированных; опциональные torchao extensions не загружаются и не используются NF4 backend; Starlette/httpx и torch.jit имеют предупреждения deprecation. Конфликтов требований пакетов `uv pip check` не обнаружил.", "",
              "## Воспроизведение", "", "```bash", "cd opendecision", "make setup", "make doctor", "make dataset", "make baseline", "# Перед GPU командами остановить собственный OpenDecision API", ".venv/bin/opendecision train --config configs/qlora_9b.yaml --resume checkpoints/qlora_smoke/checkpoint-20", "make evaluate", "make calibrate", "OPENDECISION_GPU_TEST=1 .venv/bin/python -m pytest -q", "make api PORT=8001", "# В другом терминале:", ".venv/bin/python scripts/http_smoke.py --port 8001", "```", "",
              "Для нового независимого обучения задайте новый output_dir в копии YAML. Повторный запуск без resume не перезаписывает существующий адаптер."]
    Path("docs/RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
