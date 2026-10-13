"""Диагностика без загрузки весов."""
import ctypes.util
import importlib.metadata
import json
import platform
import shutil
import subprocess
import urllib.request
from pathlib import Path

import psutil


def main():
    report = {"linux": platform.platform(), "python": platform.python_version(),
              "disk": dict(zip(["total", "used", "free"], shutil.disk_usage("."))),
              "ram": dict(psutil.virtual_memory()._asdict()),
              "libraries": {x: ctypes.util.find_library(x) for x in ["cuda", "stdc++", "gomp"]}}
    try:
        report["nvidia_smi"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        report["nvidia_error"] = str(exc)
    try:
        import torch
        report["torch_cuda"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["bf16"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if torch.cuda.is_available():
            report["cuda_capability"] = torch.cuda.get_device_capability()
    except ImportError:
        report["cuda_available"] = False
    try:
        with urllib.request.urlopen("https://huggingface.co/api/models/Qwen/Qwen3.5-9B", timeout=20) as r:
            report["huggingface"] = r.status
    except (OSError, ValueError) as exc:
        report["huggingface"] = str(exc)
    report["versions"] = {}
    for package in ["torch", "transformers", "unsloth", "unsloth_zoo", "peft", "trl",
                    "bitsandbytes", "fastapi", "pydantic", "accelerate", "triton"]:
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][package] = "MISSING"
    report["recommendations"] = []
    if not report["cuda_available"]:
        report["recommendations"].append("Проверьте nvidia-smi и установку CUDA PyTorch; 9B на CPU не запускать")
    if report["disk"]["free"] < 45 * 1024**3:
        report["recommendations"].append("Нужно минимум 45 GiB свободного диска для весов и checkpoints")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/environment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
