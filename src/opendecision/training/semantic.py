"""Смысловые сценарии с вручную заданными правилами разметки."""
import random

from .dataset import Sample


def generate_semantic(seed: int = 42) -> list[Sample]:
    rng = random.Random(seed)
    scenarios = [
        ("routing", "Забыл пароль, не могу войти.", "I forgot my password and cannot sign in.", ["Доступ к аккаунту", "Оплата", "Доставка"], ["Account access", "Billing", "Shipping"], 0),
        ("routing", "Списали дважды за одну покупку.", "I was charged twice for one purchase.", ["Доступ к аккаунту", "Оплата", "Доставка"], ["Account access", "Billing", "Shipping"], 1),
        ("routing", "Расскажите про миграцию китов.", "Tell me about whale migration.", ["Оплата", "Доставка", "Другая тема"], ["Billing", "Shipping", "Other topic"], 2),
        ("routing", "У меня проблема.", "I have a problem.", ["Оплата", "Доставка", "Нужно уточнение"], ["Billing", "Shipping", "Clarification needed"], 2),
        ("classification", "Фильм скучный, жаль времени.", "The movie was boring, a waste of time.", ["Положительный отзыв", "Отрицательный отзыв", "Нейтральный отзыв"], ["Positive review", "Negative review", "Neutral review"], 1),
        ("classification", "Посылку получил во вторник.", "The parcel arrived on Tuesday.", ["Положительный отзыв", "Отрицательный отзыв", "Нейтральный отзыв"], ["Positive review", "Negative review", "Neutral review"], 2),
        ("judging", "Задача: назвать цвет ясного дневного неба. Ответ: голубой.", "Task: name the color of a clear daytime sky. Answer: blue.", ["Ответ верен", "Ответ неверен"], ["Correct answer", "Incorrect answer"], 0),
        ("judging", "Задача: назвать любое млекопитающее. Ответ: лосось.", "Task: name any mammal. Answer: salmon.", ["Ответ верен", "Ответ неверен"], ["Correct answer", "Incorrect answer"], 1),
        ("verification", "Источник: поезд отправляется в 18:00. Утверждение: поезд отправляется в 19:00.", "Source: train departs at 18:00. Claim: it departs at 19:00.", ["Подтверждено", "Противоречит", "Недостаточно данных"], ["Supported", "Contradicted", "Insufficient evidence"], 1),
        ("verification", "Источник: поезд отправляется в 18:00. Утверждение: в поезде есть Wi-Fi.", "Source: train departs at 18:00. Claim: the train has Wi-Fi.", ["Подтверждено", "Противоречит", "Недостаточно данных"], ["Supported", "Contradicted", "Insufficient evidence"], 2),
        ("safety", "Письмо содержит просьбу сообщить код двухфакторной аутентификации незнакомцу.", "An email asks you to share your two-factor authentication code with a stranger.", ["Обычная безопасная просьба", "Подозрение на кражу доступа"], ["Routine safe request", "Suspected credential theft"], 1),
        ("safety", "Коллега просит ссылку на публичную документацию.", "A colleague asks for a link to public documentation.", ["Безопасный запрос", "Запрос секретов"], ["Safe request", "Request for secrets"], 0),
        ("risk", "Изменение протестировано, резервная копия есть, откат проверен.", "The change is tested, backups exist, rollback is verified.", ["Риск снижен мерами предосторожности", "Риск не контролируется"], ["Risk mitigated by precautions", "Risk uncontrolled"], 0),
        ("priority", "На странице настроек опечатка, функции работают.", "There is a typo on the settings page, all functions work.", ["Низкий приоритет", "Критический приоритет"], ["Low priority", "Critical priority"], 0),
        ("priority", "Все серверы недоступны, клиенты не могут работать.", "All servers are down and customers cannot work.", ["Низкий приоритет", "Критический приоритет"], ["Low priority", "Critical priority"], 1),
        ("rating", "Рубрика: 1 = без ответа, 2 = часть, 3 = всё. Нужно назвать красный и синий. Ответ: красный.", "Rubric: 1 = none, 2 = part, 3 = all. Asked to name red and blue. Answer: red.", ["1", "2", "3"], ["1", "2", "3"], 1),
        ("workflow", "Правило: сначала одобрение, затем публикация. Одобрение ещё не получено.", "Rule: approval before publication. Approval has not been received.", ["Публиковать", "Ждать одобрения"], ["Publish", "Wait for approval"], 1),
        ("workflow", "Правило: повторить при временной сетевой ошибке. Получена временная сетевая ошибка.", "Rule: retry a transient network error. A transient network error occurred.", ["Повторить", "Завершить успешно"], ["Retry", "Mark successful"], 0),
    ]
    rows = []
    for i, (task, ru, en, dru, den, target) in enumerate(scenarios):
        for lang, state, descriptions in [("ru", ru, dru), ("en", en, den)]:
            for variant in range(4):
                options = [{"id": f"semantic_{rng.randrange(10**9)}", "description": d} for d in descriptions]
                answer = options[target]["id"]
                rng.shuffle(options)
                question = (["Выберите наиболее подходящий вариант.", "Какое решение следует из описания?", "Оцените ситуацию.", "Примените смысл описаний вариантов."] if lang == "ru" else ["Select the most appropriate option.", "Which decision follows from the description?", "Assess the situation.", "Apply the option descriptions."])[variant]
                rows.append(Sample(id=f"semantic-{i}-{lang}-{variant}", task=task, state=state,
                                   question=question, options=options, answer=answer, language=lang,
                                   source="synthetic-human-labeled-v1", group_id=f"semantic-{i}"))
    return rows
