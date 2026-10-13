"""Формирование target-only loss без хрупкого поиска строк chat template."""
from opendecision.prompting import LABELS, DecisionPromptBuilder, label_sequences


def encode_sample(tokenizer, sample, max_length: int) -> dict:
    prompt = DecisionPromptBuilder().render(tokenizer, sample.request())
    prefix, labels = label_sequences(tokenizer, prompt, len(sample.options))
    index = next(i for i,o in enumerate(sample.options) if o.id == sample.answer)
    target = labels[index] + [tokenizer.eos_token_id]
    if len(prefix) + len(target) > max_length:
        raise ValueError(f"Пример {sample.id} превышает max_length; target не обрезан")
    result = {"input_ids": prefix + target, "attention_mask": [1]*(len(prefix)+len(target)),
              "labels": [-100]*len(prefix) + target}
    assert tokenizer.decode(labels[index]) == LABELS[index]
    assert all(x == -100 for x in result["labels"][:len(prefix)])
    return result
