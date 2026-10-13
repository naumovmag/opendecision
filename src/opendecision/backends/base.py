"""Минимальный контракт стратегии inference."""
from typing import Protocol

from opendecision.schemas import DecisionRequest


class ContextLengthError(ValueError):
    """Пользовательский запрос не помещается в разрешённый контекст."""


class Backend(Protocol):
    def score(self, request: DecisionRequest) -> list[float]: ...
    def metadata(self) -> dict: ...
    def close(self) -> None: ...

