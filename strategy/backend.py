"""Compute backend: PyTorch on a GPU (CUDA or Apple MPS).

The subspace strategy computes every matrix product on the GPU with PyTorch --
there is no CPU or CuPy backend. Host arrays (NumPy / memmap) are staged to the
device with ``to_device``; the multiply itself runs on ``self.dev``.

``self.xp`` is a tiny NumPy-compatible shim (``_TorchXP``) so array code written
for NumPy -- ``xp.zeros``, ``@``, ``.astype``, ``xp.concatenate``,
``xp.linalg.qr`` -- runs on torch tensors unchanged.

Standalone: no imports from the sibling `matmul` package.
"""
from __future__ import annotations

import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# PyTorch NumPy-compatibility shim
# ---------------------------------------------------------------------------
_NP_TO_TORCH = {"float16": "float16", "float32": "float32", "float64": "float64"}


def _to_torch_dtype(torch, dtype):
    """Map a NumPy dtype / torch dtype / None to a torch dtype (or None)."""
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    name = np.dtype(dtype).name
    return getattr(torch, _NP_TO_TORCH.get(name, name))


def _install_astype(torch):
    """Give torch tensors a NumPy-style ``.astype`` so shared code just works."""
    if not hasattr(torch.Tensor, "astype"):
        def astype(self, dtype, copy=False):
            out = self.to(_to_torch_dtype(torch, dtype))
            return out.clone() if copy and out is self else out
        torch.Tensor.astype = astype


class _TorchXP:
    """Minimal ``numpy``-like namespace backed by torch, bound to a device."""

    def __init__(self, torch, device):
        self.t = torch
        self.device = device
        self.linalg = torch.linalg
        self.float16, self.float32, self.float64 = (
            torch.float16, torch.float32, torch.float64,
        )

    def _dt(self, dtype):
        return _to_torch_dtype(self.t, dtype)

    def zeros(self, shape, dtype=None):
        return self.t.zeros(shape, dtype=self._dt(dtype), device=self.device)

    def empty(self, shape, dtype=None):
        return self.t.empty(shape, dtype=self._dt(dtype), device=self.device)

    def full(self, shape, fill_value, dtype=None):
        return self.t.full(shape, fill_value, dtype=self._dt(dtype), device=self.device)

    def arange(self, *args, dtype=None, **kw):
        return self.t.arange(*args, dtype=self._dt(dtype), device=self.device, **kw)

    def cos(self, x):
        return self.t.cos(x)

    def concatenate(self, tensors, axis=0):
        return self.t.cat(list(tensors), dim=axis)


def _linux_mem_available() -> int | None:
    """Linux MemAvailable in bytes, or None when it cannot be read.

    ``SC_AVPHYS_PAGES`` is MemFree on Linux: it counts only *unused* pages and
    excludes the reclaimable page cache, so it under-reports what a big
    allocation can actually get -- badly, once this process has itself written
    a disk-backed matrix (that data stays in cache and is charged against
    MemFree, though the kernel would evict it on demand). The kernel publishes
    MemAvailable for exactly this question, and it is the same quantity the
    win32 branch below already returns via ``ullAvailPhys``.
    """
    try:
        with open("/proc/meminfo", "rb") as fh:
            for line in fh:
                if line.startswith(b"MemAvailable:"):
                    return int(line.split()[1]) * 1024   # reported in kB
    except (OSError, ValueError, IndexError):
        pass
    return None


def _host_available_bytes() -> int:
    """Best-effort available physical RAM in bytes (cross-platform)."""
    avail = _linux_mem_available()
    if avail is not None and avail > 0:
        return avail
    try:
        avail = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        if avail > 0:
            return avail
    except (AttributeError, OSError, ValueError):
        pass

    if sys.platform == "win32":
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:  # noqa: BLE001
            pass

    return 8 * 1024**3  # last-resort fallback when OS queries are unavailable


# Device->host copies out of pageable memory run at roughly a seventh of the
# pinned bandwidth (measured on an RTX 3090: 268 MB in 159 ms pageable vs 21.5 ms
# pinned). Every result this engine produces -- reconstruct's (rb, n) rows and
# multiply_exact's accumulator -- leaves the device through ``to_host``, so the
# whole strategy is bounded by that copy, not by its math. Bounce large reads
# through one small, reused pinned buffer instead: a 16 MiB buffer costs ~6 ms to
# pin once and then streams at full pinned bandwidth, while pinning the entire
# result up front (268 MB -> ~94 ms) would give most of the win back.
_PINNED_CHUNK_BYTES = 16 * 1024 ** 2
# Below this the copy is too small for the bounce to pay for itself.
_PINNED_MIN_BYTES = 4 * 1024 ** 2


class Backend:
    """PyTorch GPU backend. Raises if no CUDA/MPS device is available."""

    def __init__(self, device: int = 0, verbose: bool = True):
        self.device_id = int(device)
        self.verbose = verbose
        self.kind = "torch"
        self.gpu = True
        self._pin_buf = None            # lazily allocated pinned bounce buffer

        try:
            import torch  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "CCO GPU evaluation requires PyTorch. Install the GPU extra: "
                "uv sync --extra gpu"
            ) from exc

        _install_astype(torch)
        self.torch = torch
        if torch.cuda.is_available():
            self.dev = torch.device(f"cuda:{self.device_id}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.dev = torch.device("mps")
        else:
            raise RuntimeError(
                "CCO computes on the GPU only. No CUDA or Apple-MPS device was "
                "detected by PyTorch. Run on a machine with a GPU."
            )
        self.xp = _TorchXP(torch, self.dev)

    # -- introspection -----------------------------------------------------
    @property
    def name(self) -> str:
        if self.dev.type == "cuda":
            return f"{self.torch.cuda.get_device_name(self.device_id)} (PyTorch/CUDA)"
        return "Apple GPU (PyTorch/MPS)"

    def free_compute_bytes(self) -> int:
        """Free device memory in bytes (used to size streamed row-blocks)."""
        if self.dev.type == "cuda":
            free, _total = self.torch.cuda.mem_get_info(self.device_id)
            return int(free)
        rec = getattr(self.torch.mps, "recommended_max_memory", None)
        if rec is not None:
            try:
                return int(rec())
            except Exception:  # noqa: BLE001
                pass
        return self.host_available_bytes()

    def host_available_bytes(self) -> int:
        """Available system RAM (used for out-of-core host buffering)."""
        return _host_available_bytes()

    # -- transfers ---------------------------------------------------------
    def to_device(self, host_array):
        """Copy a host (NumPy/memmap) array to the GPU as a torch tensor."""
        if self.torch.is_tensor(host_array):
            return host_array.to(self.dev)
        return self.torch.from_numpy(np.ascontiguousarray(host_array)).to(self.dev)

    def _pinned_staging(self, dtype, elems):
        """One reused pinned bounce buffer (grown on demand, never shrunk).

        Reused across calls and across row-blocks so the one-off pinning cost is
        paid once per run, not once per transfer."""
        buf = self._pin_buf
        if buf is None or buf.dtype != dtype or buf.numel() < elems:
            self._pin_buf = self.torch.empty(elems, dtype=dtype, pin_memory=True)
        return self._pin_buf[:elems]

    def to_host(self, dev_array) -> np.ndarray:
        if not self.torch.is_tensor(dev_array):
            return np.asarray(dev_array)
        t = dev_array.detach()
        if t.device.type == "cpu":
            return t.numpy()
        if t.numel() * t.element_size() < _PINNED_MIN_BYTES:
            return t.to("cpu").numpy()
        # Large read: stream it through the pinned buffer (see _PINNED_CHUNK_BYTES).
        t = t.contiguous()
        flat = t.reshape(-1)
        total = flat.numel()
        chunk = min(total, max(1, _PINNED_CHUNK_BYTES // t.element_size()))
        buf = self._pinned_staging(t.dtype, chunk)
        out = np.empty(tuple(t.shape), dtype=buf.numpy().dtype)
        out_flat = out.reshape(-1)
        for i in range(0, total, chunk):
            j = min(total, i + chunk)
            buf[: j - i].copy_(flat[i:j])
            out_flat[i:j] = buf[: j - i].numpy()
        return out

    def zeros(self, shape, dtype):
        return self.xp.zeros(shape, dtype=dtype)

    def matmul(self, a, b):
        """Matrix product for the smart (subspace) engine: ``torch.matmul`` only."""
        return self.torch.matmul(a, b)

    def synchronize(self):
        if self.dev.type == "cuda":
            self.torch.cuda.synchronize(self.device_id)
        elif self.dev.type == "mps":
            self.torch.mps.synchronize()
