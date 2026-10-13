"""Межпроцессная блокировка GPU для процессов OpenDecision."""
import fcntl
import os
from pathlib import Path


class GPULease:
    def __init__(self):
        # Блокировка на видимый набор GPU: обучение на GPU0 и API на GPU1 не конфликтуют.
        devices = os.getenv("CUDA_VISIBLE_DEVICES", "all").replace(",", "-") or "none"
        self.file = Path(f"/tmp/opendecision-gpu-{os.getuid()}-{devices}.lock").open("a")  # noqa: SIM115
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.file.close()
            raise RuntimeError(f"GPU {devices} уже используется другим процессом OpenDecision. "
                               "Остановите API/обучение или выберите другой CUDA_VISIBLE_DEVICES.") from exc

    def close(self):
        self.file.close()
