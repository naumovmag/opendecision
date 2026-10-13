import math

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from opendecision.api import create_app
from opendecision.backends.base import ContextLengthError
from opendecision.calibration import fit_temperature
from opendecision.config import Settings
from opendecision.engine import DecisionEngine
from opendecision.evaluation.robustness import transform
from opendecision.prompting import DecisionPromptBuilder, label_sequences
from opendecision.schemas import DecisionRequest
from opendecision.scoring import probabilities
from opendecision.training.prepare import generate, split


def request():
    return DecisionRequest(state="Верните деньги", question="Кто поможет?",
                           options=[{"id": "refund", "description": "Возвраты"},
                                    {"id": "delivery", "description": "Доставка"}])


class FakeBackend:
    """Только unit tests; не доказательство работы модели."""
    def score(self, request):
        return [3.0, 1.0]

    def metadata(self):
        return {"backend": "unit-test-fake"}

    def close(self):
        pass


@pytest.mark.parametrize("count", [0, 1, 27])
def test_option_count(count):
    payload = request().model_dump()
    payload["options"] = [{"id": str(i), "description": "x"} for i in range(count)]
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(payload)


def test_twenty_six_option_limit_and_label_mapping():
    payload = request().model_dump()
    payload["options"] = [{"id": f"item_{i}", "description": f"Товар {i}"}
                          for i in range(26)]
    decision = DecisionRequest.model_validate(payload)
    assert len(decision.options) == 26

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(c) for c in text]

    _, labels = label_sequences(Tokenizer(), "prefix:", len(decision.options))
    assert len(labels) == 26
    assert labels[0] == [ord("A")]
    assert labels[-1] == [ord("Z")]


def test_duplicate_ids():
    payload = request().model_dump()
    payload["options"][1]["id"] = "refund"
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(payload)


@pytest.mark.parametrize("scores", [[10000, 9999], [-10000, -10000], [0, 1, 2, 3, 4, 5, 6, 7]])
def test_normalization(scores):
    p = probabilities(scores)
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in p)
    assert sum(p) == pytest.approx(1)
    assert p.index(max(p)) == scores.index(max(scores))


def test_nonfinite_scores():
    with pytest.raises(ValueError):
        probabilities([0, float("nan")])


def test_prompt():
    messages = DecisionPromptBuilder().messages(request())
    assert "untrusted" in messages[0]["content"]
    assert "Возвраты" in messages[1]["content"]
    assert '"refund"' not in messages[1]["content"]


def test_label_mapping_multitoken():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(c) for c in text for _ in range(2)]
    prefix, labels = label_sequences(Tokenizer(), "prefix:", 3)
    assert len(prefix) == 14
    assert labels == [[65, 65], [66, 66], [67, 67]]


def test_dataset_splits():
    rows = generate()
    assert len(rows) == 2000
    assert [r.model_dump() for r in rows] == [r.model_dump() for r in generate()]
    parts = split(rows)
    groups = [{r.group_id for r in v} for v in parts.values()]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert sum(map(len, parts.values())) == len(rows)
    assert all(r.answer in {o.id for o in r.options} for r in rows)


def test_calibration():
    report = {"split": "validation", "dataset_sha256": "test", "identity": {},
              "records": [{"scores": [5, 0], "target": i % 2} for i in range(40)]}
    result = fit_temperature(report)
    assert result["temperature"] > 1
    assert result["calibrated"]["nll"] < result["raw"]["nll"]
    report["split"] = "test"
    with pytest.raises(ValueError):
        fit_temperature(report)


@pytest.mark.parametrize("kind", ["permutation", "rename", "paraphrase", "injection", "long"])
def test_transformations(kind):
    original = request()
    changed, mapping = transform(original, kind)
    assert {mapping[o.id] for o in changed.options} == {o.id for o in original.options}
    assert original.state == "Верните деньги"


def test_api():
    with TestClient(create_app(DecisionEngine(FakeBackend(), Settings()))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 200
        assert client.get("/docs").status_code == 200
        result = client.post("/v1/decisions", json=request().model_dump())
        assert result.status_code == 200
        body = result.json()
        assert body["selected"] == "refund"
        assert body["calibrated"] is False
        assert sum(body["probabilities"].values()) == pytest.approx(1)
        assert client.post("/v1/decisions", json={}).status_code == 422
        assert client.post("/v1/decisions", content="x"*1048577).status_code == 413
        batch = client.post("/v1/decisions/batch", json={"requests": [request().model_dump()]})
        assert batch.status_code == 200 and len(batch.json()) == 1


class BatchingBackend:
    """Fake backend с score_many: проверяет, что батчер собирает одновременные запросы."""
    def __init__(self, fail_state=None, delay=0.05):
        self.batches = []
        self.fail_state = fail_state
        self.delay = delay

    def score_many(self, requests):
        import time
        time.sleep(self.delay)
        self.batches.append(len(requests))
        out = []
        for r in requests:
            if r.state == self.fail_state:
                out.append(ContextLengthError("слишком длинно"))
            else:
                out.append([float(len(r.state))] + [0.0] * (len(r.options) - 1))
        return out

    def metadata(self):
        return {"backend": "unit-test-batching"}

    def close(self):
        pass


def test_batcher_groups_concurrent_requests():
    from concurrent.futures import ThreadPoolExecutor
    backend = BatchingBackend()
    engine = DecisionEngine(backend, Settings(batch_wait_ms=20))
    reqs = [request().model_copy(update={"state": "x" * (i + 1)}) for i in range(30)]
    with ThreadPoolExecutor(30) as pool:
        results = list(pool.map(engine.decide, reqs))
    assert all(r.selected == "refund" for r in results)
    assert sum(backend.batches) == 30
    assert max(backend.batches) > 1


def test_batcher_isolates_errors():
    backend = BatchingBackend(fail_state="bad", delay=0)
    engine = DecisionEngine(backend, Settings())
    good, bad = request(), request().model_copy(update={"state": "bad"})
    results = engine.decide_many([good, bad, good])
    assert results[0].selected == "refund" and results[2].selected == "refund"
    assert isinstance(results[1], ContextLengthError)
    with pytest.raises(ContextLengthError):
        engine.decide(bad)


def test_api_context_error_and_batch():
    backend = BatchingBackend(fail_state="bad", delay=0)
    with TestClient(create_app(DecisionEngine(backend, Settings()))) as client:
        bad = request().model_copy(update={"state": "bad"}).model_dump()
        assert client.post("/v1/decisions", json=bad).status_code == 422
        body = client.post("/v1/decisions/batch", json={"requests": [request().model_dump()] * 3})
        assert body.status_code == 200 and len(body.json()) == 3
        assert client.post("/v1/decisions/batch",
                           json={"requests": [request().model_dump(), bad]}).status_code == 422


def test_public_model_name():
    assert Settings().public_name == "Qwen/Qwen3.5-9B"
    assert Settings(adapter="checkpoints/qlora_pilot/adapter").public_name == "Qwen/Qwen3.5-9B+qlora_pilot"
    named = Settings(adapter="checkpoints/qlora_pilot/adapter", served_model_name="Adata-Decision-1-9B")
    with TestClient(create_app(DecisionEngine(FakeBackend(), named))) as client:
        assert client.get("/health").json()["model"] == "Adata-Decision-1-9B"
        assert client.post("/v1/decisions", json=request().model_dump()).json()["model"] == "Adata-Decision-1-9B"
