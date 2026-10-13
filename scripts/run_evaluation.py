import argparse

from opendecision.evaluation.benchmark import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--dataset", default="data/examples/independent.jsonl")
    parser.add_argument("--output", default="reports/baseline.json")
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    evaluate(args.config, args.dataset, args.output, args.split, args.limit)


if __name__ == "__main__":
    main()
