# Экспериментальное QLoRA 9B

Основной конфиг: `configs/qlora_9b.yaml`. `FastLanguageModel` из Unsloth загружает
официальные веса с `use_exact_model_name=True`, NF4, double quantization, BF16.
Unsloth импортируется раньше Transformers. VLM loader возвращает Processor;
для чисто текстового SFT используется вложенный tokenizer.

Сначала динамически находятся существующие языковые Linear-модули. Unsloth получает
их имена и `finetune_vision_layers=False`. После применения PEFT проверяется,
что trainable только LoRA, число параметров ненулевое. Полные имена сохраняются
в `target_modules.json`.

Completion содержит метку A–H и EOS. Префикс labels равен -100. Если пример длиннее
контекста, он явно записывается в `skipped_overlength`; target не обрезается.
Используется TRL SFTTrainer с уже токенизированным Dataset и собственным collator.

MemoryMonitor проверяет finite/nonzero gradients перед optimizer step и фактическое
изменение LoRA-параметров после каждого шага. В отчёте отдельно указаны число
проверенных backward/optimizer steps и суммарное изменение весов. Логи памяти:
`reports/gpu_metrics.jsonl`; итог: `checkpoints/qlora_smoke/training_report.json`.

Промежуточные checkpoints сохраняются каждые 10 шагов. Сохранение smoke и pilot
разделено. Для resume передайте `--resume checkpoints/qlora_smoke/checkpoint-20`.
Повторный независимый эксперимент должен использовать новый `output_dir`.

При OOM процесс завершается; CUDA память освобождается ОС. Максимум три изменения
конфигурации: 128 токенов; ограничение модулей до q/v; при необходимости только
q/v последних языковых слоёв. Microbatch всегда 1. CPU offload не используется.
Ошибки несовместимого API исправляются отдельно от экспериментов с памятью.

Не делайте вывод об улучшении по train loss. Сравнивайте baseline и adapter тем же
Transformers NF4 backend на одинаковом test и ручном OOD наборе. Калибровку обучайте
только на validation. После обучения перед запуском API тренировочный процесс
должен завершиться.
