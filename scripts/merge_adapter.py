"""Слияние LoRA-адаптера с базой Qwen3.5 в самодостаточную BF16-модель.

Адаптер, обученный на NF4-базе (QLoRA), при слиянии с точной BF16-базой даёт
заметно другие ответы. Флаг --dequantized-base сливает адаптер с той же
NF4-базой, развёрнутой в BF16, — это воспроизводит поведение NF4+adapter.
Расхождение в любом случае нужно измерить отдельно (run_evaluation).
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--revision", default="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    parser.add_argument("--adapter", default="checkpoints/qlora_pilot/adapter")
    parser.add_argument("--output", default="models/opendecision-qwen3.5-9b-pilot-merged")
    parser.add_argument("--dequantized-base", action="store_true",
                        help="База = NF4 (как при QLoRA-обучении и в API), развёрнутая в BF16")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"{output} уже существует; укажите другой --output")
    kwargs = {"revision": args.revision, "dtype": torch.bfloat16, "device_map": {"": 0}}
    if args.dequantized_base:
        # Те же параметры квантизации, что в TransformersBackend и train.py.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, **kwargs)
    if args.dequantized_base:
        model = model.dequantize()
        model = model.to(torch.bfloat16)
        if getattr(model.config, "quantization_config", None):
            del model.config.quantization_config
        if hasattr(model, "is_quantized"):
            model.is_quantized = False
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("После слияния остались LoRA-параметры")
    model.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(args.model, revision=args.revision).save_pretrained(output)
    for name in ["chat_template.jinja"]:
        source = Path(args.adapter) / name
        if source.exists() and not (output / name).exists():
            shutil.copy(source, output / name)
    adapter_config = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
    (output / "opendecision_merge.json").write_text(json.dumps({
        "base_model": args.model, "base_revision": args.revision, "adapter": args.adapter,
        "lora_r": adapter_config.get("r"), "lora_alpha": adapter_config.get("lora_alpha"),
        "target_modules": adapter_config.get("target_modules"), "merged_dtype": "bfloat16",
        "base_weights": "nf4-dequantized" if args.dequantized_base else "bf16-original",
    }, indent=2))
    print(f"Сохранено: {output}")


if __name__ == "__main__":
    main()
