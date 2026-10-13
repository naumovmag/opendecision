import json
import os
from pathlib import Path

import pytest

from opendecision.config import Settings
from opendecision.engine import create_engine
from opendecision.training.dataset import DatasetRepository


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("OPENDECISION_GPU_TEST") != "1", reason="Явно включите реальную GPU проверку")
def test_real_qwen_scoring():
    engine = create_engine(Settings.load("configs/inference.yaml"))
    try:
        row = DatasetRepository.read("data/examples/independent.jsonl")[0]
        result = engine.decide(row.request())
        assert sum(result.probabilities.values()) == pytest.approx(1)
        assert len(engine.backend.last_labels) == len(row.options)
        from opendecision.schemas import Option
        req = row.request()
        req.options.extend(Option(id=f"extra_{i}", description=f"Другая категория {i}") for i in range(5))
        eight = engine.decide(req)
        assert len(eight.probabilities) == 8
        assert len(engine.backend.last_labels) == 8
    finally:
        engine.backend.close()


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("OPENDECISION_GPU_TEST") != "1", reason="Реальная проверка adapter")
def test_trained_adapter():
    path = Path("checkpoints/qlora_smoke")
    report = json.loads((path / "training_report.json").read_text())
    assert report["evidence"]["optimizer_steps_with_weight_change"] == 30
    assert report["evidence"]["finite_gradient_checks"] == 30
    assert (path / "adapter/adapter_model.safetensors").stat().st_size > 0
    engine = create_engine(Settings.load("configs/inference-adapter.yaml"))
    try:
        row = DatasetRepository.read("data/examples/independent.jsonl")[0]
        response = engine.decide(row.request())
        assert response.selected in {o.id for o in row.options}
        assert sum(response.probabilities.values()) == pytest.approx(1)
    finally:
        engine.backend.close()


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("OPENDECISION_GPU_TEST") != "1", reason="Кеш префикса против полного прогона")
def test_prefix_cache_batch_matches_full_forward():
    import torch

    from opendecision.backends.transformers_backend import TransformersBackend
    from opendecision.prompting import label_sequences
    from opendecision.schemas import DecisionRequest

    backend = TransformersBackend(Settings.load(os.getenv("OPENDECISION_GPU_CONFIG", "configs/inference.yaml")))
    try:
        rows = DatasetRepository.read("data/examples/independent.jsonl")
        options = rows[0].request().options
        # Общий префикс у всех запросов + запросы со своими вариантами в одном батче.
        requests = [DecisionRequest(state=r.state, question=rows[0].question, options=options) for r in rows]
        requests += [r.request() for r in rows]
        batched = backend.score_many(requests)
        assert backend.stats["prefix_hits"] + backend.stats["prefix_misses"] < len(requests)
        for req, got in zip(requests, batched):
            assert not isinstance(got, Exception)
            prompt = backend.builder.render(backend.tokenizer, req)
            full, labels = label_sequences(backend.tokenizer, prompt, len(req.options))
            with torch.inference_mode():
                ref = backend.model(input_ids=torch.tensor([full], device="cuda"), use_cache=False,
                                    logits_to_keep=1).logits[0, -1, [x[0] for x in labels]].float().cpu()
            assert float((torch.tensor(got) - ref).abs().max()) < 1.0
            top2 = ref.topk(2).values
            if float(top2[0] - top2[1]) > 1.0:  # при почти равных scores выбор решает шум BF16
                assert int(torch.tensor(got).argmax()) == int(ref.argmax())
        assert backend.score_many(requests[:3]) and backend.stats["prefix_hits"] > 0
    finally:
        backend.close()
