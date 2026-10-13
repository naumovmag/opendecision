"""Клиент локального сервиса."""
import json
from pathlib import Path

import httpx

if __name__ == "__main__":
    payload = json.loads(Path("data/examples/request.json").read_text())
    response = httpx.post("http://127.0.0.1:8000/v1/decisions", json=payload, timeout=120)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
