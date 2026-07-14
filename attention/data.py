"""Synthetic Q/K/V generation for the attention playground."""
from __future__ import annotations

from .spec import AttentionSpec


def _torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "The attention playground requires PyTorch. Install the GPU extra: "
            "uv sync --extra gpu"
        ) from exc
    return torch


def resolve_device(device: str):
    torch = _torch()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def torch_dtype(dtype: str):
    torch = _torch()
    return {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }[dtype]


def generate_qkv(spec: AttentionSpec, device=None):
    """Return synthetic Q/K/V tensors for one attention benchmark run."""
    torch = _torch()
    dev = device if device is not None else resolve_device(spec.device)
    # The RNG must live on the SAME device as the tensors it seeds -- including
    # the index. dev.type drops the index, so a --device cuda:1 run built the
    # generator on "cuda" (= cuda:0) and torch.randn(generator=..., device=cuda:1)
    # then raised a device mismatch. MPS has no device-side generator -> use CPU.
    gen_device = "cpu" if dev.type == "mps" else dev
    gen = torch.Generator(device=gen_device).manual_seed(spec.seed)
    dtype = torch_dtype(spec.dtype)
    shape = (spec.batch, spec.heads, spec.seq, spec.dim)
    q = torch.randn(shape, generator=gen, device=dev, dtype=dtype)
    k = torch.randn(shape, generator=gen, device=dev, dtype=dtype)
    v = torch.randn(shape, generator=gen, device=dev, dtype=dtype)
    return q, k, v
