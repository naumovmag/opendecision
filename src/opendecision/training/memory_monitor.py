"""Реальные метрики памяти, градиентов и optimizer steps."""
import json
import time
from pathlib import Path

import psutil
import torch
from transformers import TrainerCallback


class MemoryMonitor(TrainerCallback):
    def __init__(self, model, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.perf_counter()
        self.gradient_checks = 0
        self.nonzero_gradient_steps = 0
        self.changed_steps = 0
        self.before = {n: p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        self.initial = {n: v.clone() for n,v in self.before.items()}
        self.nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except (ImportError, RuntimeError):
            pass

    def on_step_begin(self, args, state, control, **kwargs):
        self.start = time.perf_counter()

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError("Отсутствуют градиенты либо найден NaN/Inf")
        self.gradient_checks += 1
        if any(torch.count_nonzero(g).item() > 0 for g in gradients):
            self.nonzero_gradient_steps += 1
        else:
            raise RuntimeError("Все LoRA градиенты равны нулю")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        delta = 0.0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                now = parameter.detach().float().cpu()
                delta += float((now - self.before[name]).square().sum())
                self.before[name] = now.clone()
        if delta <= 0:
            raise RuntimeError("Optimizer step не изменил LoRA веса")
        self.changed_steps += 1
        record = {"experiment": args.output_dir, "step": state.global_step, "seconds": time.perf_counter()-self.start,
                  "allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
                  "peak_allocated": torch.cuda.max_memory_allocated(),
                  "peak_reserved": torch.cuda.max_memory_reserved(),
                  "ram_used": psutil.virtual_memory().used, "lora_delta_squared": delta}
        if self.nvml:
            record["gpu_utilization"] = self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            record["gpu_device_used"] = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def evidence(self, model) -> dict:
        delta = sum(float((p.detach().float().cpu()-self.initial[n]).square().sum())
                    for n,p in model.named_parameters() if p.requires_grad)
        return {"finite_gradient_checks": self.gradient_checks,
                "nonzero_gradient_steps": self.nonzero_gradient_steps,
                "optimizer_steps_with_weight_change": self.changed_steps,
                "total_lora_delta_squared": delta}
