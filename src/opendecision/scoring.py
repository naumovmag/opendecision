"""Нормализация полных scores допустимых кандидатов."""
import numpy as np


def probabilities(scores: list[float], temperature: float = 1.0) -> list[float]:
    a = np.asarray(scores, dtype=np.float64)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Scores должны быть конечным вектором минимум из двух элементов")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Температура должна быть положительной и конечной")
    a = (a - a.max()) / temperature
    p = np.exp(a)
    return (p / p.sum()).tolist()

