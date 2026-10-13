"""Нагрузочный тест: N одновременных запросов, время до последнего ответа."""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from opendecision.training.dataset import DatasetRepository


def scenarios() -> dict[str, list[dict]]:
    assistants = json.loads(Path("data/examples/adata_assistants.json").read_text())
    rows = []
    for f in ["data/processed/validation.jsonl", "data/processed/test.jsonl",
              "data/examples/independent.jsonl"]:
        rows += DatasetRepository.read(f)
    states = list(dict.fromkeys(r.state for r in rows))
    routing = [{"state": s, "question": "Какой ассистент должен обработать обращение?",
                "options": assistants} for s in states]
    unique, seen = [], set()
    for r in rows:
        key = (r.question, tuple(o.description for o in r.options))
        if key not in seen:
            seen.add(key)
            unique.append(r.request().model_dump())
    return {"routing": routing, "unique": unique}


async def burst(client: httpx.AsyncClient, url: str, payloads: list[dict]) -> dict:
    async def one(p):
        t = time.perf_counter()
        r = await client.post(url, json=p)
        return r.status_code, (time.perf_counter() - t) * 1000
    start = time.perf_counter()
    results = await asyncio.gather(*(one(p) for p in payloads))
    wall = (time.perf_counter() - start) * 1000
    lat = sorted(ms for _, ms in results)
    return {"n": len(payloads), "all_done_ms": round(wall), "p50_ms": round(lat[len(lat) // 2]),
            "p95_ms": round(lat[int(len(lat) * 0.95) - 1]), "max_ms": round(lat[-1]),
            "errors": sum(code != 200 for code, _ in results)}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8012/v1/decisions")
    parser.add_argument("--sizes", default="1,10,20,30,40,50")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    data = scenarios()
    limits = httpx.Limits(max_connections=256, max_keepalive_connections=256)
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:
        await client.post(args.url, json=data["routing"][0])  # прогрев
        for name, pool in data.items():
            for n in [int(x) for x in args.sizes.split(",")]:
                runs = []
                for k in range(args.repeats):
                    offset = (k * n) % max(1, len(pool) - n)
                    runs.append(await burst(client, args.url, pool[offset:offset + n]))
                best = min(runs, key=lambda r: r["all_done_ms"])
                worst = max(runs, key=lambda r: r["all_done_ms"])
                print(json.dumps({"scenario": name, "concurrent": n,
                                  "all_done_ms_best": best["all_done_ms"],
                                  "all_done_ms_worst": worst["all_done_ms"],
                                  "p50_ms": best["p50_ms"], "p95_ms": best["p95_ms"],
                                  "errors": sum(r["errors"] for r in runs)}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
