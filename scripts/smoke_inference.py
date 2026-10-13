from opendecision.config import Settings
from opendecision.engine import create_engine
from opendecision.training.dataset import DatasetRepository

if __name__ == "__main__":
    engine = create_engine(Settings.load("configs/inference.yaml"))
    try:
        for row in DatasetRepository.read("data/examples/independent.jsonl")[:10]:
            print(engine.decide(row.request()).model_dump_json(indent=2), flush=True)
    finally:
        engine.backend.close()
