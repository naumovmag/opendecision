"""Независимая проверка adapter, маски и teacher-forced loss обычным Transformers."""
import json
from pathlib import Path

import torch
from transformers import DataCollatorForSeq2Seq

from opendecision.config import Settings
from opendecision.engine import create_engine
from opendecision.training.dataset import DatasetRepository
from opendecision.training.validation import encode_sample


def main():
    engine = create_engine(Settings.load("configs/inference-adapter.yaml"))
    backend = engine.backend
    rows = DatasetRepository.read("data/processed/validation.jsonl")[:8]
    results = []
    try:
        collator = DataCollatorForSeq2Seq(backend.tokenizer, label_pad_token_id=-100)
        for row in rows:
            encoded = encode_sample(backend.tokenizer, row, 512)
            batch = collator([encoded])
            indices = [i for i,x in enumerate(encoded["labels"]) if x != -100]
            assert len(indices) == 2
            assert batch["labels"][0, indices].tolist() == [encoded["input_ids"][i] for i in indices]
            with torch.inference_mode():
                # Получаем logits отдельно, loss считаем явно со сдвигом на один токен.
                logits = backend.model(input_ids=batch["input_ids"].cuda(), use_cache=False,
                                       logits_to_keep=3).logits[0, :-1].float()
                targets = batch["labels"][0, indices].cuda()
                loss = torch.nn.functional.cross_entropy(logits, targets)
            result = engine.decide(row.request())
            results.append({"id": row.id, "target_tokens": targets.tolist(),
                            "decoded_target": backend.tokenizer.decode(targets.tolist()),
                            "prompt_tokens_masked": indices[0], "supervised_tokens": 2,
                            "independent_cross_entropy": float(loss),
                            "correct": result.selected == row.answer})
        data = {"status": "PASS", "base_model": backend.config.model,
                "adapter": backend.config.adapter, "records": results,
                "quantization": backend.model.config.quantization_config}
        if hasattr(data["quantization"], "to_dict"):
            data["quantization"] = data["quantization"].to_dict()
        Path("reports/adapter-audit.json").write_text(json.dumps(data, indent=2))
        print(json.dumps(data, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
