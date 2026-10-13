"""Метрики для переменного числа вариантов; F1 по семантическим ответам."""
import numpy as np
from sklearn.metrics import f1_score

from opendecision.scoring import probabilities


def metrics(records: list[dict], temperature: float = 1.0) -> dict:
    if not records:
        raise ValueError("Пустой набор оценки")
    correct, confidences, losses, briers, truth, pred = [], [], [], [], [], []
    for row in records:
        p = np.asarray(probabilities(row["scores"], temperature))
        y = row["target"]
        best = int(p.argmax())
        correct.append(best == y)
        confidences.append(float(p[best]))
        losses.append(-np.log(max(float(p[y]), 1e-300)))
        onehot = np.eye(len(p))[y]
        briers.append(float(((p-onehot)**2).sum()))
        names = row.get("semantic_labels", [str(i) for i in range(len(p))])
        truth.append(names[y])
        pred.append(names[best])
    bins = []
    ece = 0.0
    for i in range(10):
        indices = [j for j,c in enumerate(confidences) if i/10 <= c and (c < (i+1)/10 or i == 9)]
        if indices:
            acc = float(np.mean([correct[j] for j in indices]))
            conf = float(np.mean([confidences[j] for j in indices]))
            ece += len(indices)/len(records)*abs(acc-conf)
            bins.append({"lower": i/10, "upper": (i+1)/10, "count": len(indices), "accuracy": acc, "confidence": conf})
    lat = [r["latency_ms"] for r in records if "latency_ms" in r]
    return {"count": len(records), "accuracy": float(np.mean(correct)),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "nll": float(np.mean(losses)), "brier": float(np.mean(briers)), "ece": ece,
            "reliability_bins": bins,
            "p50_latency_ms": float(np.percentile(lat, 50)) if lat else None,
            "p95_latency_ms": float(np.percentile(lat, 95)) if lat else None,
            "throughput_per_second": 1000*len(lat)/sum(lat) if lat and sum(lat) else None}
