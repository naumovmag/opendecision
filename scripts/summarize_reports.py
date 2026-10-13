"""Дополняет отчёты метриками из уже измеренных scores, без новых inference."""
import json
from pathlib import Path

from opendecision.evaluation.compare import compare
from opendecision.evaluation.metrics import metrics
from opendecision.scoring import probabilities
from opendecision.training.dataset import DatasetRepository


def main():
    for path in ["reports/baseline.json", "reports/baseline-test.json",
                 "reports/fine-tuned.json", "reports/fine-tuned-test.json",
                 "reports/pilot.json", "reports/pilot-test.json"]:
        if not Path(path).exists():
            continue
        report = json.loads(Path(path).read_text())
        samples = {r.id: r for r in DatasetRepository.read(report["dataset"])}
        for record in report["records"]:
            record["language"] = samples[record["id"]].language
            record["probabilities"] = probabilities(record["scores"])
        report["by_language"] = {lang: metrics([r for r in report["records"] if r["language"] == lang])
                                 for lang in ("ru", "en")}
        report["errors"] = [r["id"] for r in report["records"] if max(range(len(r["scores"])), key=r["scores"].__getitem__) != r["target"]]
        Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    compare([p for p in ["reports/baseline-test.json", "reports/fine-tuned-test.json", "reports/pilot-test.json"] if Path(p).exists()], "reports/comparison-test.md")
    compare([p for p in ["reports/baseline.json", "reports/fine-tuned.json", "reports/pilot.json"] if Path(p).exists()], "reports/comparison-ood.md")


if __name__ == "__main__":
    main()
