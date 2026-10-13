"""Реальный scoring Qwen3.5: батч с кешем общего префикса либо полная likelihood метки.

Промпт устроен как system + {"question", "options", "state"}: всё до значения state
совпадает у запросов с одинаковыми вопросом и вариантами. Префикс считается один раз
(LRU-кеш состояний Gated DeltaNet и KV полного внимания), хвосты со state
прогоняются одним батчем с паддингом справа: реальные токены паддинга не видят.
"""
import copy
import gc
import importlib.metadata
from collections import OrderedDict

from opendecision.config import Settings
from opendecision.prompting import DecisionPromptBuilder, label_sequences
from opendecision.schemas import DecisionRequest

from .base import ContextLengthError

# Символ из Private Use Area: не встречается в обычных данных и не экранируется json.dumps.
STATE_MARKER = "STATE"


class TransformersBackend:
    def __init__(self, config: Settings):
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("GPU не обнаружена: загрузка большой модели на CPU запрещена")
        from .resources import GPULease
        self.lease = GPULease()
        self.config = config
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
        kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}, "revision": config.revision}
        if config.quantize:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(config.model, **kwargs)
        if config.adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, config.adapter)
        self.model.eval()
        # LoRA внедрена в модули, поэтому текстовая часть базовой модели уже с адаптером.
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        self.language_model, self.lm_head = base.model.language_model, base.lm_head
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        self.builder = DecisionPromptBuilder()
        self.last_labels = []
        self.prefix_caches: OrderedDict[tuple[int, ...], object] = OrderedDict()
        self.prefix_tokens: OrderedDict[str, list[int]] = OrderedDict()
        self.label_tokens: dict[int, list[list[int]]] = {}
        self.stats = {"requests": 0, "batches": 0, "prefix_hits": 0, "prefix_misses": 0}
        # Хвост шаблона после user-сообщения одинаков у всех промптов, поэтому токены меток
        # A–Z можно проверить один раз на число вариантов, а не токенизировать каждый промпт.
        probe = DecisionRequest(state=STATE_MARKER, question="q",
                                options=[{"id": "a", "description": "a"}, {"id": "b", "description": "b"}])
        rendered = self.builder.render(self.tokenizer, probe)
        user = self.builder.messages(probe)[1]["content"]
        self.template_tail = rendered[rendered.rindex(user) + len(user):]
        torch.cuda.reset_peak_memory_stats()

    def _labels(self, prompt: str, count: int) -> list[list[int]]:
        if not prompt.endswith(self.template_tail):
            return label_sequences(self.tokenizer, prompt, count)[1]
        if count not in self.label_tokens:
            self.label_tokens[count] = label_sequences(self.tokenizer, prompt, count)[1]
        return self.label_tokens[count]

    def _prefix_ids(self, prefix_text: str) -> list[int]:
        if prefix_text in self.prefix_tokens:
            self.prefix_tokens.move_to_end(prefix_text)
            return self.prefix_tokens[prefix_text]
        ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        self.prefix_tokens[prefix_text] = ids
        while len(self.prefix_tokens) > 1024:
            self.prefix_tokens.popitem(last=False)
        return ids

    def _encode(self, request: DecisionRequest) -> tuple[list[int], list[int], list[list[int]]]:
        prompt = self.builder.render(self.tokenizer, request)
        full = self.tokenizer.encode(prompt, add_special_tokens=False)
        labels = self._labels(prompt, len(request.options))
        if len(full) + max(map(len, labels)) > self.config.max_tokens:
            raise ContextLengthError(f"Запрос превышает лимит {self.config.max_tokens} токенов")
        marked = self.builder.render(self.tokenizer, request.model_copy(update={"state": STATE_MARKER}))
        prefix_text = marked[:marked.index(STATE_MARKER)]
        prefix = list(self._prefix_ids(prefix_text)) if prompt.startswith(prefix_text) else []
        # Граница BPE могла склеить кавычку с началом state: укорачиваем до общего префикса.
        while prefix and full[:len(prefix)] != prefix:
            prefix = prefix[:-1]
        if len(prefix) >= len(full):
            prefix = full[:-1]
        return full, prefix, labels

    def score(self, request: DecisionRequest) -> list[float]:
        result = self.score_many([request])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def score_many(self, requests: list[DecisionRequest]) -> list[list[float] | Exception]:
        """Scores для каждого запроса; ошибка одного запроса возвращается на его месте."""
        results: list = [None] * len(requests)
        groups: OrderedDict[tuple[int, ...], list] = OrderedDict()
        for i, request in enumerate(requests):
            try:
                full, prefix, labels = self._encode(request)
                self.last_labels = labels
                if all(len(x) == 1 for x in labels):
                    groups.setdefault(tuple(prefix), []).append((i, full, [x[0] for x in labels]))
                else:
                    results[i] = self._score_multitoken(full, labels)
            except Exception as exc:  # noqa: BLE001 - ошибка относится только к этому запросу
                results[i] = exc
        self.stats["requests"] += len(requests)
        for prefix, items in groups.items():
            try:
                cache = self._prefix_cache(prefix)
                for chunk in self._chunks(items, len(prefix)):
                    for (i, _, _), scores in zip(chunk, self._score_suffixes(cache, len(prefix), chunk)):
                        results[i] = scores
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, self.torch.cuda.OutOfMemoryError):
                    self.prefix_caches.clear()
                    self.torch.cuda.empty_cache()
                for i, _, _ in items:
                    if results[i] is None:
                        results[i] = exc
        return results

    def warmup(self) -> float:
        """Прогон типичных форм батча: JIT Triton-ядер до первого реального запроса."""
        import time
        start = time.perf_counter()
        options = [{"id": f"o{i}", "description": f"Вариант {i}: " + "описание категории " * 12}
                   for i in range(6)]
        sizes = sorted({s for s in (1, 2, 4, 8, 16, 32, 48, 64) if s <= self.config.max_batch_size}
                       | {self.config.max_batch_size})
        for size in sizes:
            for words in (3, 20, 80):
                requests = [DecisionRequest(state=f"Обращение {i}: " + "текст " * words,
                                            question="Прогрев", options=options) for i in range(size)]
                for outcome in self.score_many(requests):
                    if isinstance(outcome, Exception):
                        raise outcome
        self.prefix_caches.clear()
        self.stats = dict.fromkeys(self.stats, 0)
        return time.perf_counter() - start

    def _chunks(self, items: list, prefix_len: int):
        width = max(len(full) - prefix_len for _, full, _ in items)
        size = max(1, min(self.config.max_batch_size,
                          self.config.max_batch_tokens // (prefix_len + width)))
        for start in range(0, len(items), size):
            yield items[start:start + size]

    def _prefix_cache(self, prefix: tuple[int, ...]):
        if not prefix:
            return None
        if prefix in self.prefix_caches:
            self.prefix_caches.move_to_end(prefix)
            self.stats["prefix_hits"] += 1
            return self.prefix_caches[prefix]
        self.stats["prefix_misses"] += 1
        with self.torch.inference_mode():
            cache = self.language_model(input_ids=self.torch.tensor([prefix], device="cuda"),
                                        use_cache=True).past_key_values
        if self.config.prefix_cache_size:
            self.prefix_caches[prefix] = cache
            while len(self.prefix_caches) > self.config.prefix_cache_size:
                self.prefix_caches.popitem(last=False)
        return cache

    def _expand(self, cache, n: int):
        """Копия кеша префикса, размноженная на n строк батча (исходный кеш не меняется)."""
        cache = copy.deepcopy(cache)
        for layer in cache.layers:
            for name in ("keys", "values"):
                tensor = getattr(layer, name, None)
                if isinstance(tensor, self.torch.Tensor):
                    setattr(layer, name, tensor.repeat_interleave(n, dim=0))
            for name in ("conv_states", "recurrent_states"):
                states = getattr(layer, name, None)
                if isinstance(states, dict):
                    for key, tensor in states.items():
                        if isinstance(tensor, self.torch.Tensor):
                            states[key] = tensor.repeat_interleave(n, dim=0)
        return cache

    def _score_suffixes(self, cache, prefix_len: int, items: list) -> list[list[float]]:
        torch = self.torch
        n = len(items)
        width = max(len(full) - prefix_len for _, full, _ in items)
        ids = torch.full((n, width), self.pad_id, dtype=torch.long)
        last = []
        for row, (_, full, _) in enumerate(items):
            suffix = full[prefix_len:]
            ids[row, :len(suffix)] = torch.tensor(suffix)
            last.append(len(suffix) - 1)
        with torch.inference_mode():
            past = self._expand(cache, n) if cache is not None else None
            hidden = self.language_model(input_ids=ids.cuda(), past_key_values=past,
                                         use_cache=past is not None).last_hidden_state
            picked = hidden[torch.arange(n, device="cuda"), torch.tensor(last, device="cuda")]
            scores = []
            for row, (_, _, label_ids) in enumerate(items):
                # lm_head только по строкам меток: без logits по всему словарю.
                scores.append((picked[row:row + 1] @ self.lm_head.weight[label_ids].T)
                              .float()[0].cpu().tolist())
        self.stats["batches"] += 1
        return scores

    def _score_multitoken(self, full: list[int], labels: list[list[int]]) -> list[float]:
        torch = self.torch
        scores = []
        with torch.inference_mode():
            for label in labels:
                inputs = torch.tensor([full + label], device="cuda")
                logits = self.model(input_ids=inputs, use_cache=False,
                                    logits_to_keep=len(label) + 1).logits[0, :-1].float()
                targets = torch.tensor(label, device="cuda")
                scores.append(float(logits.log_softmax(-1).gather(1, targets[:, None]).sum()))
        return scores

    def metadata(self) -> dict:
        return {
            "served_model_name": self.config.public_name,
            "model": self.config.model, "revision": self.config.revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "tokenizer_revision": self.tokenizer.init_kwargs.get("_commit_hash"),
            "backend": "transformers-bnb4" if self.config.quantize else "transformers-bf16",
            "adapter": self.config.adapter, "label_tokens": self.last_labels,
            "batching": {"max_batch_size": self.config.max_batch_size,
                         "batch_wait_ms": self.config.batch_wait_ms,
                         "prefix_cache_size": self.config.prefix_cache_size,
                         "prefix_cache_entries": len(self.prefix_caches), **self.stats},
            "versions": {p: importlib.metadata.version(p) for p in
                         ["torch", "transformers", "bitsandbytes", "peft"]},
            "gpu_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(),
        }

    def close(self) -> None:
        self.prefix_caches.clear()
        del self.model, self.language_model, self.lm_head
        gc.collect()
        self.torch.cuda.empty_cache()
        self.lease.close()
