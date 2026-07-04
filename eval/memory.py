"""Peak GPU-memory measurement for a block of compute — the "pick of VRAM".

`MemoryProbe` reports the *peak incremental* device memory a computation used,
over the allocation already live at entry:

    * CUDA: exact, from the torch caching allocator
      (``reset_peak_memory_stats`` + ``max_memory_allocated``) — no sampling.
    * MPS : best-effort, sampling ``torch.mps.current_allocated_memory`` on a
      background thread.

Compute is GPU-only (see ``strategy.backend`` / ``matmul.backend``), so there is
no host-memory path here.
"""
from __future__ import annotations

import threading


class MemoryProbe:
    """Context manager exposing peak incremental device memory as ``peak_bytes``.

        probe = MemoryProbe(backend)
        with probe:
            ...compute...
        print(probe.peak_bytes)
    """

    def __init__(self, backend, interval: float = 0.002):
        self._backend = backend
        self._torch = backend.torch
        self._dev = backend.dev
        self._interval = interval
        self._cuda = self._dev.type == "cuda"
        self._base = 0
        self._peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- MPS sampling ------------------------------------------------------
    def _mps_used(self) -> int:
        fn = getattr(self._torch.mps, "current_allocated_memory", None)
        try:
            return int(fn()) if fn is not None else 0
        except Exception:  # noqa: BLE001
            return 0

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            used = self._mps_used()
            if used > self._peak:
                self._peak = used
            self._stop.wait(self._interval)

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "MemoryProbe":
        self._backend.synchronize()
        if self._cuda:
            self._base = int(self._torch.cuda.memory_allocated(self._dev))
            self._torch.cuda.reset_peak_memory_stats(self._dev)
        else:
            self._base = self._mps_used()
            self._peak = self._base
            self._stop.clear()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._backend.synchronize()
        if self._cuda:
            self._peak = int(self._torch.cuda.max_memory_allocated(self._dev))
        else:
            used = self._mps_used()
            if used > self._peak:
                self._peak = used
            self._stop.set()
            if self._thread is not None:
                self._thread.join()

    @property
    def peak_bytes(self) -> int:
        """Peak device memory above the entry baseline (never negative)."""
        return max(0, self._peak - self._base)
