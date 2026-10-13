"""Temperature scaling исключительно на validation."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from .config import Settings
from .evaluation.metrics import metrics


def fit_temperature(report: dict) -> dict:
    if report.get("split") != "validation":
        raise ValueError("Калибровку можно обучать только на validation split")
    records = report["records"]
    result = minimize_scalar(lambda log_t: metrics(records, float(np.exp(log_t)))["nll"],
                             bounds=(-3, 3), method="bounded")
    temperature = float(np.exp(result.x))
    from .scoring import probabilities
    candidates = []
    for threshold in np.linspace(0.5, 0.99, 50):
        accepted = []
        for row in records:
            p = probabilities(row["scores"], temperature)
            if max(p) >= threshold:
                accepted.append(int(np.argmax(p)) == row["target"])
        if len(accepted) >= 10 and np.mean(accepted) >= 0.9:
            candidates.append((len(accepted), float(threshold), float(np.mean(accepted))))
    selection = max(candidates, default=(0, 1.0, None))
    return {"temperature": temperature, "experimental": len(records) < 500,
            "resolved_revision": report.get("metadata", {}).get("resolved_revision"),
            "confidence_threshold": selection[1], "validation_accepted_count": selection[0],
            "validation_accepted_accuracy": selection[2],
            "sample_count": len(records), "split": "validation",
            "dataset_sha256": report["dataset_sha256"], "identity": report["identity"],
            "raw": metrics(records), "calibrated": metrics(records, temperature)}


def load_calibration(path: str, settings: Settings, resolved_revision: str | None = None) -> dict:
    data = json.loads(Path(path).read_text())
    identity = {"model": settings.model, "revision": settings.revision,
                "quantize": settings.quantize, "adapter": settings.adapter}
    if data["identity"] != identity or data.get("split") != "validation":
        raise ValueError("Калибровка не соответствует модели или split")
    if data.get("resolved_revision") != resolved_revision:
        raise ValueError("Commit модели не совпадает с калибровкой")
    if not np.isfinite(data["temperature"]) or data["temperature"] <= 0:
        raise ValueError("Некорректная температура")
    return data
