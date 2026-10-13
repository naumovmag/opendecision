from pathlib import Path

import pytest
import yaml

from opendecision.training.dataset import Sample
from opendecision.training.validation import encode_sample


class CharacterTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "PROMPT:"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def sample():
    return Sample(id="test", task="routing", language="ru", source="unit-test", group_id="case",
                  state="Отмена заказа", question="Кто нужен?",
                  options=[{"id": "a", "description": "Доставка"}, {"id": "b", "description": "Отмена"}], answer="b")


def test_completion_mask():
    encoded = encode_sample(CharacterTokenizer(), sample(), 32)
    assert encoded["labels"] == [-100]*7 + [ord("B"), 0]
    assert encoded["input_ids"][-2:] == [ord("B"), 0]


def test_target_never_truncated():
    with pytest.raises(ValueError):
        encode_sample(CharacterTokenizer(), sample(), 8)


def test_qlora_configuration():
    config = yaml.safe_load(Path("configs/qlora_9b.yaml").read_text())
    assert config["model"] == "Qwen/Qwen3.5-9B"
    assert config["quantize"] is True
    assert config["r"] == 8
    assert config["max_steps"] == 30
    assert config["max_seq_length"] == 256


def test_invalid_dataset_answer():
    payload = sample().model_dump()
    payload["answer"] = "missing"
    with pytest.raises(ValueError):
        Sample.model_validate(payload)
