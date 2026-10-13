"""Скачивание официальных весов с фиксацией commit и проверкой диска."""
import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--revision", default=None, help="Точный Hugging Face commit")
    args = parser.parse_args()
    info = HfApi().model_info(args.model, revision=args.revision, files_metadata=True)
    size = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))
    if shutil.disk_usage(Path.home()).free < size + 15 * 1024**3:
        raise RuntimeError(f"Недостаточно диска: нужно {size / 1024**3 + 15:.1f} GiB")
    path = snapshot_download(args.model, revision=info.sha, allow_patterns=[
        "*.json", "*.safetensors", "*.jinja", "tokenizer*", "vocab*", "merges*", "README.md"])
    out = {"model": args.model, "revision": info.sha, "path": path, "weight_bytes": size,
           "license": (info.card_data or {}).get("license")}
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/download-{args.model.split('/')[-1]}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
