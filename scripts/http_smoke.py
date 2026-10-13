"""Проверка реально запущенного HTTP сервера, без подмены backend."""
import argparse
import json
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", default="reports/api-smoke.json")
    args = parser.parse_args()
    payload = json.loads(Path("data/examples/request.json").read_text())
    with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=120) as client:
        health = client.get("/health")
        health.raise_for_status()
        models = client.get("/v1/models")
        models.raise_for_status()
        assert client.get("/docs").status_code == 200
        response = client.post("/v1/decisions", json=payload)
        response.raise_for_status()
        result = response.json()
        assert abs(sum(result["probabilities"].values())-1) < 1e-6
        assert result["selected"] in {o["id"] for o in payload["options"]}
        batch = client.post("/v1/decisions/batch", json={"requests": [payload, payload]})
        batch.raise_for_status()
        assert len(batch.json()) == 2
        invalid = client.post("/v1/decisions", json={"state": "missing fields"})
        assert invalid.status_code == 422
    report = {"status": "PASS", "port": args.port, "health": health.json(), "models": models.json(),
              "response": result, "batch_count": 2, "invalid_status": 422, "swagger_status": 200}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
