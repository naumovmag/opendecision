"""DI и динамический батчинг запросов к единственной модели."""
import queue
import threading
import time
from concurrent.futures import Future

from .backends.base import Backend
from .config import Settings
from .schemas import DecisionRequest, DecisionResponse
from .scoring import probabilities


class DecisionEngine:
    def __init__(self, backend: Backend, settings: Settings):
        self.backend = backend
        self.settings = settings
        self.temperature = 1.0
        self.calibrated = False
        self.calibration_experimental = None
        if settings.calibration:
            from .calibration import load_calibration
            calibration = load_calibration(settings.calibration, settings,
                                           backend.metadata().get("resolved_revision"))
            self.temperature = calibration["temperature"]
            self.settings.confidence_threshold = calibration["confidence_threshold"]
            self.calibrated = True
            self.calibration_experimental = calibration["experimental"]
        # Модель на GPU не потокобезопасна: к backend обращается только поток батчера.
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._start_lock = threading.Lock()

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        result = self.decide_many([request])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def submit(self, requests: list[DecisionRequest]) -> list[Future]:
        """Ставит запросы в очередь батчера; каждый Future завершится ответом или ошибкой."""
        self._ensure_worker()
        futures = []
        for request in requests:
            future: Future = Future()
            self._queue.put((request, time.perf_counter(), future))
            futures.append(future)
        return futures

    def decide_many(self, requests: list[DecisionRequest]) -> list[DecisionResponse | Exception]:
        """Синхронно ждёт все ответы; ошибки возвращаются на местах запросов."""
        results = []
        for future in self.submit(requests):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - ошибка относится к одному запросу
                results.append(exc)
        return results

    def _ensure_worker(self) -> None:
        with self._start_lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="opendecision-batcher",
                                                daemon=True)
                self._worker.start()

    def _run(self) -> None:
        wait = self.settings.batch_wait_ms / 1000
        while True:
            batch = [self._queue.get()]
            deadline = time.perf_counter() + wait
            while len(batch) < self.settings.max_batch_size:
                timeout = deadline - time.perf_counter()
                try:
                    batch.append(self._queue.get(timeout=timeout) if timeout > 0
                                 else self._queue.get_nowait())
                except queue.Empty:
                    break
            self._process(batch)

    def _process(self, batch: list) -> None:
        requests = [request for request, _, _ in batch]
        try:
            if hasattr(self.backend, "score_many"):
                outcomes = self.backend.score_many(requests)
            else:
                outcomes = []
                for request in requests:
                    try:
                        outcomes.append(self.backend.score(request))
                    except Exception as exc:  # noqa: BLE001
                        outcomes.append(exc)
        except Exception as exc:  # noqa: BLE001 - сбой всего батча
            outcomes = [exc] * len(batch)
        for (request, start, future), outcome in zip(batch, outcomes):
            if isinstance(outcome, Exception):
                future.set_exception(outcome)
                continue
            try:
                future.set_result(self._response(request, outcome, start))
            except Exception as exc:  # noqa: BLE001
                future.set_exception(exc)

    def _response(self, request: DecisionRequest, scores: list[float], start: float) -> DecisionResponse:
        if len(scores) != len(request.options):
            raise RuntimeError("Backend вернул неверное число scores")
        p = probabilities(scores, self.temperature)
        best = max(range(len(p)), key=p.__getitem__)
        threshold = request.confidence_threshold
        if threshold is None:
            threshold = self.settings.confidence_threshold
        return DecisionResponse(
            model=self.settings.public_name, selected=request.options[best].id, confidence=p[best],
            probabilities={o.id: v for o, v in zip(request.options, p)},
            calibrated=self.calibrated, requires_review=p[best] < threshold,
            calibration_experimental=self.calibration_experimental,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def create_engine(settings: Settings) -> DecisionEngine:
    from .backends.transformers_backend import TransformersBackend
    return DecisionEngine(TransformersBackend(settings), settings)
