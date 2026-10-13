"""JSONL repository и строгая валидация происхождения примеров."""
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from opendecision.schemas import DecisionRequest


class Sample(DecisionRequest):
    id: str
    task: Literal["classification", "routing", "judging", "verification", "safety", "risk",
                  "priority", "rating", "workflow"]
    answer: str
    language: Literal["ru", "en", "kk", "mixed"]
    source: str
    group_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_answer(self) -> Self:
        if self.answer not in {o.id for o in self.options}:
            raise ValueError("answer отсутствует в options")
        return self

    def request(self) -> DecisionRequest:
        return DecisionRequest.model_validate(self.model_dump(include=set(DecisionRequest.model_fields)))


class DatasetRepository:
    @staticmethod
    def read(path: str | Path) -> list[Sample]:
        rows = [Sample.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if len({x.id for x in rows}) != len(rows):
            raise ValueError("Повторяющиеся sample IDs")
        return rows

    @staticmethod
    def write(path: str | Path, rows: list[Sample]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r.model_dump(), ensure_ascii=False) + "\n" for r in rows))

