"""Контракты динамических решений."""
import math

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class Option(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128, pattern=r"^[\w.:-]+$")
    description: str = Field(min_length=1, max_length=2000)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str = Field(min_length=1, max_length=24000)
    question: str = Field(min_length=1, max_length=2000)
    options: list[Option] = Field(min_length=2, max_length=26)
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        if len({o.id for o in self.options}) != len(self.options):
            raise ValueError("option IDs должны быть уникальны")
        return self


class DecisionResponse(BaseModel):
    model: str
    selected: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    calibrated: bool
    calibration_experimental: bool | None = None
    requires_review: bool
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def valid_probabilities(self) -> Self:
        p = list(self.probabilities.values())
        if not p or any(not math.isfinite(x) or not 0 <= x <= 1 for x in p):
            raise ValueError("Некорректные вероятности")
        if abs(sum(p) - 1) > 1e-6 or self.selected not in self.probabilities:
            raise ValueError("Некорректное распределение")
        if abs(self.confidence - self.probabilities[self.selected]) > 1e-6:
            raise ValueError("confidence не соответствует выбранному ответу")
        return self


class BatchRequest(BaseModel):
    requests: list[DecisionRequest] = Field(min_length=1, max_length=16)

