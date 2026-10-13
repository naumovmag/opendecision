"""Сверка автономного opendecision_minimal.py с движком OpenDecision на одном наборе."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

from opendecision.training.dataset import DatasetRepository


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimal", default="models/hf_pilot/opendecision_minimal.py")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True, help="Отчёт run_evaluation того же набора")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("minimal", args.minimal)
    minimal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(minimal)
    start = time.perf_counter()
    tok, model = minimal.load(args.model, args.quantize)
    load_seconds = time.perf_counter() - start
    reference = {r["id"]: r for r in json.loads(Path(args.reference).read_text())["records"]}
    rows = DatasetRepository.read(args.dataset)
    max_dp, disagree, latencies = 0.0, [], []
    torch.cuda.reset_peak_memory_stats()
    for row in rows:
        t = time.perf_counter()
        out = minimal.decide(tok, model, row.state, row.question,
                             [o.model_dump() for o in row.options])
        latencies.append((time.perf_counter() - t) * 1000)
        ref = reference[row.id]["probabilities"]
        mine = [out["probabilities"][o.id] for o in row.options]
        max_dp = max(max_dp, max(abs(a - b) for a, b in zip(mine, ref)))
        if max(range(len(ref)), key=ref.__getitem__) != max(range(len(mine)), key=mine.__getitem__):
            disagree.append(row.id)
    latencies.sort()
    print(json.dumps({
        "count": len(rows), "selection_disagreements": disagree, "max_abs_prob_diff": max_dp,
        "load_seconds": round(load_seconds, 1),
        "p50_ms": round(latencies[len(latencies) // 2], 1),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
        "gpu_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "gpu_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "quantize": args.quantize, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }, ensure_ascii=False, indent=2))
    sys.exit(1 if disagree else 0)


if __name__ == "__main__":
    main()
