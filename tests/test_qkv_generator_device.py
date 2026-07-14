"""CPU-safe test: the QKV generator's RNG must sit on the tensors' device.

Follows the attention playground's torch-optional convention (returns early when
torch is absent, so it is a no-op in the CPU CI). On a machine with >1 CUDA
device it is a real regression guard for the generator/tensor device mismatch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

from attention import AttentionSpec

if torch is not None:
    from attention import generate_qkv


def _skip_if_no_torch():
    if torch is None:
        print("SKIP  torch not installed")
        return True
    return False


def test_generator_lives_on_tensor_device():
    """generate_qkv must build its RNG on the SAME device (including index) as
    the Q/K/V it seeds; otherwise a --device cuda:N (N > 0) run raises a device
    mismatch from torch.randn(generator=...)."""
    if _skip_if_no_torch():
        return
    if torch.cuda.is_available():
        # Highest-indexed CUDA device: exactly where a bare-"cuda" (= cuda:0)
        # generator mismatches the requested tensor device.
        dev = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
    else:
        dev = torch.device("cpu")
    spec = AttentionSpec(batch=1, heads=1, seq=8, dim=4, dtype="fp32", seed=0)
    q, k, v = generate_qkv(spec, device=dev)
    assert q.device == dev and k.device == dev and v.device == dev


if __name__ == "__main__":
    ok = True
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_"):
            try:
                _fn()
                print(f"PASS  {_name}")
            except AssertionError as e:
                ok = False
                print(f"FAIL  {_name}: {e}")
    sys.exit(0 if ok else 1)
