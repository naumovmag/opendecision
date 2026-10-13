"""Проверяемые настройки сервиса."""
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "Qwen/Qwen3.5-9B"
    revision: str = "main"
    quantize: bool = True
    max_tokens: int = Field(default=2048, ge=128, le=16384)
    confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    adapter: str | None = None
    calibration: str | None = None
    # Имя модели в ответах API; по умолчанию «база+адаптер», чтобы не выдавать базу за дообученную.
    served_model_name: str | None = None
    # Динамический батчинг: запросы, пришедшие за batch_wait_ms, считаются одним проходом.
    max_batch_size: int = Field(default=64, ge=1, le=512)
    batch_wait_ms: float = Field(default=5.0, ge=0, le=1000)
    max_batch_tokens: int = Field(default=65536, ge=1024)
    # Сколько общих префиксов (system + question + options) держать на GPU, ~70 MB каждый.
    prefix_cache_size: int = Field(default=16, ge=0, le=1024)
    # Прогрев Triton-ядер при старте HTTP-сервиса (первый батч новой формы иначе +0,5 с).
    warmup: bool = True

    @property
    def public_name(self) -> str:
        if self.served_model_name:
            return self.served_model_name
        if self.adapter:
            return f"{self.model}+{Path(self.adapter.rstrip('/')).parent.name or Path(self.adapter).name}"
        return self.model

    @classmethod
    def load(cls, path: str | Path) -> "Settings":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

