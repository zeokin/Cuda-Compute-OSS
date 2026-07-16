"""Pin the landing page's cost-axis claim to CONTRIBUTING's definition.

`index/index.html`'s stat band advertises how many cost axes a submission must
beat. CONTRIBUTING.md is the authority:

    Cost is `time complexity`, `latency`, and `VRAM usage`. Accuracy is the
    bounded Frobenius score against the exact product.

so there are THREE cost axes, and accuracy is a separate floor gate -- not a
cost. The distinction is load-bearing: the "never averaged / every axis must
beat exact" dominance rule applies to the cost axes only, while accuracy IS
deliberately traded (`accuracy x (1/VRAM) x (1/latency)` -- "a cheaper method
that gives up a little accuracy is a genuine win ... the accuracy factor
discounts exactly what you traded away"). The page had counted accuracy as a
fourth "cost axis ... never traded off", contradicting both halves of the rule
-- and its own "Tier 1 - Accuracy floor" section.

Derives the expected count from CONTRIBUTING itself, so the page can't drift.
Pure parsing; no GPU needed.

Run:  python tests/test_landing_page_cost_axes.py
"""
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PAGE = os.path.join(_ROOT, "index", "index.html")
_CONTRIBUTING = os.path.join(_ROOT, "CONTRIBUTING.md")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _documented_cost_axes() -> list:
    """The backticked items in CONTRIBUTING's `Cost is ...` sentence."""
    m = re.search(r"Cost is (.+?)\.\s", _read(_CONTRIBUTING), re.S)
    assert m, "CONTRIBUTING.md no longer defines `Cost is ...`"
    return re.findall(r"`([^`]+)`", m.group(1))


def _page_cost_axis_count() -> int:
    m = re.search(
        r'<div class="n">(\d+)</div><div class="l">Cost axes[^<]*</div>', _read(_PAGE)
    )
    assert m, "no cost-axes stat found on the landing page"
    return int(m.group(1))


def test_contributing_defines_three_cost_axes_excluding_accuracy():
    axes = _documented_cost_axes()
    assert len(axes) == 3, f"expected 3 cost axes, CONTRIBUTING lists {axes}"
    assert not any("accuracy" in a.lower() for a in axes), (
        f"accuracy is a floor gate, not a cost axis; got {axes}")


def test_landing_page_cost_axis_count_matches_contributing():
    assert _page_cost_axis_count() == len(_documented_cost_axes())


def test_landing_page_does_not_call_accuracy_a_never_traded_cost_axis():
    """Accuracy is traded off by design (within the floor), so the cost-axis
    stat must not sweep it in under a 'never traded off' banner."""
    page = _read(_PAGE)
    m = re.search(r'<div class="n">\d+</div><div class="l">(Cost axes[^<]*)</div>', page)
    assert m, "no cost-axes stat found on the landing page"
    assert "never traded off" not in m.group(1).lower()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
