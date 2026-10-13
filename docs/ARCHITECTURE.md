# Архитектура

`schemas` определяет динамический запрос. `DecisionPromptBuilder` сериализует
описания и недоверенное состояние в JSON внутри user message. Системная инструкция
и chat template одинаковы для SFT и inference.

`Backend` — Strategy; `TransformersBackend` адаптирует Qwen3.5 VLM к текстовому
candidate scoring. `create_engine` — Factory, `DecisionEngine` получает backend
через DI, нормализует scores, применяет calibrator и review threshold.
Модель создаётся в FastAPI lifespan, используется одна на процесс. К GPU обращается
только поток динамического батчера: HTTP-обработчики ставят запросы в очередь и ждут
Future, батчер собирает всё, что пришло за `batch_wait_ms`, и считает одним проходом.
Запросы с общим префиксом (system + question + options) группируются: префикс
прогоняется один раз, его KV-кеш и recurrent state Gated DeltaNet размножаются на
строки батча (LRU-кеш на GPU), заново считается только хвост со `state`. Хвосты
выравниваются только правым padding — padding в середине последовательности испортил
бы состояние DeltaNet. Файловая блокировка на каждую GPU предотвращает конкурирующие
собственные процессы обучения и сервиса.

Qwen3.5 сочетает 3 слоя Gated DeltaNet и 1 full attention. В full attention
существуют q/k/v/o_proj, в MLP — gate/up/down_proj. Конкретные найденные пути
сохраняются перед обучением. Vision-слои не включаются в адаптер.

`DatasetRepository` отвечает за JSONL и валидацию; `prepare` — генерация,
дедупликация и групповые splits. `evaluation` сохраняет scores, хеш данных,
список sample IDs, версии и метрики. `compare` запрещает сравнивать разные samples.

Нормализованные вероятности условны на набор кандидатов. Temperature scaling
имеет собственную identity модели/adapter/quantization и не переносится автоматически
между адаптерами. При экспериментальной калибровке это указано в её JSON.
