"""Экспериментальный QLoRA 9B: реальные gradients и optimizer steps."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml


def load_init_adapter(model, trainable: dict, cfg: dict) -> dict:
    """Стартовые веса LoRA из готового адаптера: каждый обучаемый тензор должен быть загружен."""
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    source = Path(cfg["init_adapter"])
    init_cfg = json.loads((source / "adapter_config.json").read_text())
    if (init_cfg["r"], init_cfg["lora_alpha"]) != (cfg["r"], cfg["lora_alpha"]):
        raise RuntimeError("r/lora_alpha стартового адаптера не совпадают с конфигом")
    state = load_file(source / "adapter_model.safetensors")
    result = set_peft_model_state_dict(model, state)
    if getattr(result, "unexpected_keys", None):
        raise RuntimeError(f"Лишние ключи стартового адаптера: {result.unexpected_keys[:3]}")
    params = dict(model.named_parameters())
    covered = set()
    for key, value in state.items():
        name = key.replace(".lora_A.weight", ".lora_A.default.weight") \
                  .replace(".lora_B.weight", ".lora_B.default.weight")
        param = params.get(name)
        if param is None or not torch.equal(param.detach().float().cpu(), value.float()):
            raise RuntimeError(f"Тензор стартового адаптера не загружен: {key}")
        covered.add(name)
    missing = set(trainable) - covered
    if missing:
        raise RuntimeError(f"Обучаемые параметры без стартовых весов: {sorted(missing)[:3]}")
    return {"init_adapter": str(source), "tensors_loaded": len(covered),
            "trainable_tensors": len(trainable)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qlora_9b.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    # Unsloth должен импортироваться раньше transformers/peft.
    from unsloth import FastLanguageModel  # isort: skip

    import torch
    from datasets import Dataset
    from transformers import BitsAndBytesConfig, DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    from opendecision.backends.resources import GPULease

    from .dataset import DatasetRepository
    from .memory_monitor import MemoryMonitor
    from .validation import encode_sample

    if not torch.cuda.is_available():
        raise RuntimeError("Для обучения требуется GPU")
    quantize = cfg.get("quantize", True)
    # BF16 LoRA 9B: ~19 GiB весов + активации; на 16 GiB не помещается, на 32 GiB (RTX 5090) — да.
    if not quantize and "9B" in cfg["model"] and torch.cuda.get_device_properties(0).total_memory < 28*1024**3:
        raise RuntimeError("BF16 обучение 9B запрещено на GPU менее 28 GiB")
    lease = GPULease()
    quantization = None
    if quantize:
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_compute_dtype=torch.bfloat16)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"], max_seq_length=cfg["max_seq_length"],
        dtype=torch.bfloat16, load_in_4bit=quantize, load_in_16bit=not quantize,
        full_finetuning=False, use_exact_model_name=True, revision=cfg.get("revision", "main"),
        quantization_config=quantization, offload_embedding=False,
    )
    # VLM loader возвращает Processor; для текстового SFT нужен его tokenizer.
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    quantization_evidence = getattr(model.config, "quantization_config", None)
    if hasattr(quantization_evidence, "to_dict"):
        quantization_evidence = quantization_evidence.to_dict()
    if quantize and (not quantization_evidence or not quantization_evidence.get("load_in_4bit")
                     or quantization_evidence.get("bnb_4bit_quant_type") != "nf4"):
        raise RuntimeError("Модель загружена без требуемой NF4 квантизации")
    if not quantize:
        if quantization_evidence:
            raise RuntimeError("Ожидалась BF16 модель, но загружена квантованная")
        quantization_evidence = {"dtype": "bfloat16", "load_in_4bit": False}
    candidates = cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    targets = [name for name,module in model.named_modules()
               if name.split(".")[-1] in candidates and "visual" not in name
               and "vision" not in name and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise RuntimeError("Не найдены поддерживаемые языковые Linear-модули")
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    if (Path(cfg["output_dir"]) / "adapter/adapter_model.safetensors").exists() and not args.resume:
        raise RuntimeError("Адаптер уже существует. Для нового эксперимента укажите другой output_dir; для продолжения --resume.")
    Path(cfg["output_dir"] + "/target_modules.json").write_text(json.dumps(targets, indent=2))
    model = FastLanguageModel.get_peft_model(
        model, r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0,
        target_modules=sorted({name.split(".")[-1] for name in targets}),
        finetune_vision_layers=False, finetune_language_layers=True,
        bias="none", use_gradient_checkpointing="unsloth", random_state=cfg["seed"],
    )
    trainable = {n: p for n,p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in n or "visual" in n or "vision" in n for n in trainable):
        raise RuntimeError("Нарушено замораживание базовой модели/vision encoder")
    init_evidence = None
    if cfg.get("init_adapter"):
        init_evidence = load_init_adapter(model, trainable, cfg)
        print(json.dumps(init_evidence), flush=True)
    trainable_count, total_count = model.get_nb_trainable_parameters()
    counts = {"total_parameters": total_count, "trainable_parameters": trainable_count,
              "trainable_percentage": 100*trainable_count/total_count}
    print(json.dumps(counts), flush=True)
    parts = {}
    skipped = {}
    for split in ["train", "validation"]:
        rows = DatasetRepository.read(cfg.get("datasets", {}).get(split, f"data/processed/{split}.jsonl"))
        if split == "validation" and cfg.get("eval_samples", 24):
            rows = rows[:cfg.get("eval_samples", 24)]
        encoded = []
        skipped[split] = []
        for sample in rows:
            try:
                encoded.append(encode_sample(tokenizer, sample, cfg["max_seq_length"]))
            except ValueError as exc:
                if "превышает max_length" not in str(exc):
                    raise
                skipped[split].append(sample.id)
        if len(encoded) < (32 if split == "train" else 4):
            raise RuntimeError("Недостаточно примеров, помещающихся в контекст")
        parts[split] = Dataset.from_list(encoded)
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    training_args = SFTConfig(
        output_dir=cfg["output_dir"], max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"], optim=cfg["optim"], bf16=True,
        warmup_steps=0,
        logging_steps=1, save_steps=cfg.get("eval_steps", 10), save_total_limit=2,
        eval_strategy="steps", eval_steps=cfg.get("eval_steps", 10), seed=cfg["seed"],
        # Лучший по eval_loss чекпоинт не удаляется ротацией и загружается в конце.
        load_best_model_at_end=cfg.get("load_best", False),
        metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", max_length=cfg["max_seq_length"],
        dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False,
        dataloader_num_workers=0, packing=False,
    )
    monitor = MemoryMonitor(model, "reports/gpu_metrics.jsonl")
    trainer = SFTTrainer(model=model, processing_class=tokenizer,
                         train_dataset=parts["train"], eval_dataset=parts["validation"],
                         data_collator=collator, args=training_args, callbacks=[monitor])
    torch.cuda.reset_peak_memory_stats()
    try:
        result = trainer.train(resume_from_checkpoint=args.resume)
    except torch.cuda.OutOfMemoryError:
        Path(cfg["output_dir"] + "/failure.json").write_text(json.dumps({
            "error": "CUDA OutOfMemoryError", "config": cfg,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved()}))
        raise
    if not math.isfinite(result.training_loss):
        raise RuntimeError("Нечисловой loss")
    trainer.save_model(cfg["output_dir"] + "/adapter")
    tokenizer.save_pretrained(cfg["output_dir"] + "/adapter")
    report = {"config": cfg, "metrics": result.metrics, "history": trainer.state.log_history,
              "quantization": quantization_evidence,
              "parameter_counts": counts, "evidence": monitor.evidence(model),
              "dataset_hashes": {name: hashlib.sha256(Path(cfg.get("datasets", {}).get(name, f"data/processed/{name}.jsonl")).read_bytes()).hexdigest()
                                 for name in ["train", "validation", "test"]},
              "skipped_overlength": skipped, "target_modules": targets,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "adapter": cfg["output_dir"] + "/adapter", "target_mask_checked": True,
              "init_adapter": init_evidence,
              "best_checkpoint": trainer.state.best_model_checkpoint,
              "best_eval_loss": trainer.state.best_metric}
    Path(cfg["output_dir"] + "/training_report.json").write_text(json.dumps(report, indent=2))
    Path("reports/training.json").write_text(json.dumps(report, indent=2))
    lease.close()


if __name__ == "__main__":
    main()
