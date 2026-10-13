"""Повторяемая оценка с сырыми scores и отпечатком набора."""
import hashlib
import json
import time
from pathlib import Path

from opendecision.config import Settings
from opendecision.engine import create_engine
from opendecision.scoring import probabilities
from opendecision.training.dataset import DatasetRepository

from .metrics import metrics


def evaluate(config: str, dataset: str, output: str, split: str = "test", limit: int = 0):
    settings = Settings.load(config)
    rows = DatasetRepository.read(dataset)
    if limit:
        rows = rows[:limit]
    engine = create_engine(settings)
    records = []
    try:
        for row in rows:
            start = time.perf_counter()
            scores = engine.backend.score(row.request())
            records.append({"id": row.id, "task": row.task, "language": row.language, "scores": scores,
                            "probabilities": probabilities(scores),
                            "target": next(i for i,o in enumerate(row.options) if o.id == row.answer),
                            "semantic_labels": [o.description for o in row.options],
                            "latency_ms": (time.perf_counter()-start)*1000})
            print(f"{len(records)}/{len(rows)} {row.id}", flush=True)
        report = {"split": split, "dataset": dataset,
                  "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
                  "sample_ids": [r.id for r in rows],
                  "identity": settings.model_dump(include={"model", "revision", "quantize", "adapter"}),
                  "metadata": engine.backend.metadata(), "metrics": metrics(records), "records": records}
        report["by_language"] = {lang: metrics([r for r in records if r["language"] == lang])
                                 for lang in {r["language"] for r in records}}
        report["errors"] = [r["id"] for r in records if max(range(len(r["scores"])), key=r["scores"].__getitem__) != r["target"]]
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        Path(output).with_suffix(".md").write_text("# Реальная оценка\n\n```json\n" + json.dumps(report["metrics"], indent=2) + "\n```\n")
        return report
    finally:
        engine.backend.close()
