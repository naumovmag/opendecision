"""Явный импорт небольших открытых JSONL с provenance и allowlist лицензий."""
import hashlib
import json
from pathlib import Path

from .dataset import DatasetRepository

ALLOWED_LICENSES = {"apache-2.0", "mit", "cc0-1.0", "cc-by-4.0", "bsd-3-clause"}


def import_dataset(path: str, manifest_path: str, output: str):
    manifest = json.loads(Path(manifest_path).read_text())
    for key in ["source_url", "license", "license_url", "revision", "sha256"]:
        if not manifest.get(key):
            raise ValueError(f"В provenance отсутствует {key}")
    if manifest["license"].lower() not in ALLOWED_LICENSES:
        raise ValueError("Лицензия требует отдельной проверки")
    raw = Path(path).read_bytes()
    if len(raw) > 50*1024**2:
        raise ValueError("Импорт ограничен 50 MiB")
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("Хеш источника не совпадает")
    rows = DatasetRepository.read(path)
    DatasetRepository.write(output, rows)
    Path(output + ".provenance.json").write_text(json.dumps(manifest, indent=2))


def teacher_label(input_path: str, output_path: str, engine, threshold: float = 0.9):
    """Локальная teacher-модель; низкая уверенность отправляется на ручную проверку."""
    accepted, review = [], []
    for row in DatasetRepository.read(input_path):
        result = engine.decide(row.request())
        if result.confidence < threshold:
            review.append({"sample": row.model_dump(), "decision": result.model_dump()})
            continue
        accepted.append(row.model_copy(update={"answer": result.selected,
                                               "source": f"local-teacher:{engine.settings.model}"}))
    DatasetRepository.write(output_path, accepted)
    Path(output_path + ".review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2))
