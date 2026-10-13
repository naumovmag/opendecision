"""Сравнение только одинаковых примеров."""
import json
from pathlib import Path


def compare(paths: list[str], output: str = "reports/comparison.md"):
    reports = [json.loads(Path(p).read_text()) for p in paths]
    if len({(r["dataset_sha256"], tuple(r["sample_ids"])) for r in reports}) != 1:
        raise ValueError("Нельзя сравнивать разные тестовые наборы")
    lines = ["| Модель | Accuracy | Macro-F1 | NLL | Brier | ECE | P50 ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        name = r["identity"]["model"] + (" + LoRA" if r["identity"]["adapter"] else " baseline")
        m = r["metrics"]
        lines.append("| " + name + " | " + " | ".join(f"{m[k]:.4f}" for k in ["accuracy", "macro_f1", "nll", "brier", "ece", "p50_latency_ms"]) + " |")
    Path(output).write_text("\n".join(lines) + "\n")
