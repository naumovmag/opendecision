"""CLI для JSON запросов и локальных экспериментов."""
import os
import runpy
from pathlib import Path

import typer

from .config import Settings
from .engine import create_engine
from .schemas import DecisionRequest

app = typer.Typer(help="Локальная Decision Model")


@app.command()
def doctor():
    runpy.run_path("scripts/doctor.py", run_name="__main__")


@app.command()
def classify(input: Path, config: str = "configs/inference.yaml"):
    engine = create_engine(Settings.load(config))
    try:
        print(engine.decide(DecisionRequest.model_validate_json(input.read_text())).model_dump_json(indent=2))
    finally:
        engine.backend.close()


@app.command()
def evaluate(config: str = "configs/inference.yaml", dataset: str = "data/examples/independent.jsonl",
             output: str = "reports/evaluation.json"):
    from .evaluation.benchmark import evaluate as run
    run(config, dataset, output)


@app.command()
def benchmark(config: str = "configs/inference.yaml", dataset: str = "data/examples/independent.jsonl",
              output: str = "reports/benchmark.json"):
    evaluate(config, dataset, output)


@app.command()
def train(config: str = "configs/qlora_9b.yaml", resume: str | None = None):
    import subprocess
    import sys
    command = [sys.executable, "-m", "opendecision.training.train", "--config", config]
    if resume:
        command += ["--resume", resume]
    subprocess.run(command, check=True)


@app.command()
def serve(port: int = 8000, host: str = "127.0.0.1"):
    import uvicorn
    uvicorn.run("opendecision.api:app", host=host or os.getenv("OPENDECISION_HOST", "127.0.0.1"),
                port=port, workers=1)
