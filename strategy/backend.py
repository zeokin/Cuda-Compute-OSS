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


class Backend:
    """PyTorch GPU backend. Raises if no CUDA/MPS device is available."""

    def __init__(self, device: int = 0, verbose: bool = True):
        self.device_id = int(device)
        self.verbose = verbose
        self.kind = "torch"
        self.gpu = True

        try:
            import torch  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "CCO requires PyTorch. Install it (see requirements.txt): "
                "pip install torch"
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
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            return 8 * 1024**3  # conservative fallback: assume 8 GiB

    # -- transfers ---------------------------------------------------------
    def to_device(self, host_array):
        """Copy a host (NumPy/memmap) array to the GPU as a torch tensor."""
        if self.torch.is_tensor(host_array):
            return host_array.to(self.dev)
        return self.torch.from_numpy(np.ascontiguousarray(host_array)).to(self.dev)

    def to_host(self, dev_array) -> np.ndarray:
        if self.torch.is_tensor(dev_array):
            return dev_array.detach().to("cpu").numpy()
        return np.asarray(dev_array)

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
