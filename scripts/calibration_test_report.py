"""Применяет готовый validation calibrator к test без повторного подбора."""
import json
from pathlib import Path

from opendecision.evaluation.metrics import metrics

if __name__ == "__main__":
    calibration = json.loads(Path("checkpoints/calibration.json").read_text())
    test = json.loads(Path("reports/fine-tuned-test.json").read_text())
    if calibration["identity"] != test["identity"]:
        raise ValueError("Разные модели для test и calibration")
    if calibration["dataset_sha256"] == test["dataset_sha256"]:
        raise ValueError("Validation совпадает с test")
    result = {"split": "test", "temperature": calibration["temperature"],
              "experimental": calibration["experimental"],
              "raw": metrics(test["records"]),
              "calibrated": metrics(test["records"], calibration["temperature"])}
    Path("reports/calibration-test.json").write_text(json.dumps(result, indent=2))
