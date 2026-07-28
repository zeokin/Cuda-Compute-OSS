"""Automatic local-attention blocking must remain memory bounded."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from attention import hybrid


def test_wide_window_does_not_expand_automatic_query_block(monkeypatch):
    query_heights = []
    real_matmul = torch.matmul

    def record_matmul(left, right):
        if left.ndim == 4 and right.ndim == 4:
            query_heights.append(int(left.shape[-2]))
        return real_matmul(left, right)

    monkeypatch.setattr(torch, "matmul", record_matmul)
    q = torch.randn(1, 1, 128, 4)
    hybrid.local_window_attention(q, q, q, window=96)

    assert query_heights
    assert max(query_heights) <= hybrid._DEFAULT_LOCAL_QUERY_BLOCK


def test_smaller_blocks_preserve_local_attention_result():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 96, 8)
    k = torch.randn(1, 2, 96, 8)
    v = torch.randn(1, 2, 96, 8)

    single_block = hybrid.local_window_attention(
        q, k, v, window=80, block_size=96
    )
    automatic = hybrid.local_window_attention(q, k, v, window=80)

    assert torch.allclose(automatic, single_block, atol=1e-5, rtol=1e-5)
