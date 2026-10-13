import argparse
import json
from pathlib import Path

from opendecision.calibration import fit_temperature

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/validation.json")
    parser.add_argument("--output", default="reports/calibration.json")
    args = parser.parse_args()
    result = fit_temperature(json.loads(Path(args.input).read_text()))
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
