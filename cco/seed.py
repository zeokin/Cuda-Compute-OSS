"""
cco/seed.py — derive the per-PR input seed for the canonical rerun (Step 4).

Anti-overfit mechanism. Instead of a fixed `seed=42`, the canonical rerun seeds input generation
from the **PR HEAD SHA**, which the miner cannot predict (they don't know their own future commit
hash, and the rerun is maintainer-driven). A kernel that memorizes/hardcodes outputs for known
inputs then fails correctness, because the oracle re-derives the expected output on the same
unpredictable inputs. Self-scoring uses the published fixed seed (SELF_SCORE_SEED) so miners can
iterate locally; only the canonical rerun uses the PR-HEAD seed.

benchmark.py takes the seed as an integer (`--seed`); this module derives that integer from a SHA
on the maintainer side. The whole 5-stage run is threaded with one base seed — every
`input_generator` call site uses it, so no path is left at the fixed 42 (which would otherwise
allow partial memorization of that path's inputs).

Usage:
    uv run --no-project python cco/seed.py --self-test
    uv run --no-project python cco/seed.py <pr-head-sha>     # prints the integer seed
"""

from __future__ import annotations

SELF_SCORE_SEED = 42      # published; miners iterate locally with this
DEFAULT_HEX_CHARS = 16    # mirrors cco.config.json scoring.seed.hex_chars (64 bits of the SHA)

_HEX = set("0123456789abcdef")


def seed_from_sha(sha: str, hex_chars: int = DEFAULT_HEX_CHARS) -> int:
    """Map a git commit SHA (hex string) to a non-negative 63-bit int (safe for torch.manual_seed)."""
    s = sha.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    if not s or any(c not in _HEX for c in s):
        raise ValueError(f"not a hex SHA: {sha!r}")
    return int(s[:hex_chars], 16) & ((1 << 63) - 1)


def _self_test() -> int:
    failures = 0

    def check(cond, label):
        nonlocal failures
        print(("ok   " if cond else "FAIL ") + label)
        if not cond:
            failures += 1

    sha = "40a34aae1f2b3c4d5e6f70819a2b3c4d5e6f7081"
    s1 = seed_from_sha(sha)
    check(isinstance(s1, int) and 0 <= s1 < (1 << 63), "seed is a non-negative 63-bit int")
    check(seed_from_sha(sha) == s1, "deterministic for the same SHA")
    check(seed_from_sha("0x" + sha) == s1, "0x prefix tolerated")
    check(seed_from_sha("dead" + "beef" * 9) != s1, "different SHA -> different seed")
    check(seed_from_sha(sha) != SELF_SCORE_SEED, "PR-HEAD seed differs from the self-score seed")
    try:
        seed_from_sha("not-a-sha!!")
        check(False, "rejects non-hex input")
    except ValueError:
        check(True, "rejects non-hex input")

    print("-" * 50)
    print("SELF-TEST PASSED" if not failures else f"SELF-TEST FAILED: {failures} case(s)")
    return 1 if failures else 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Derive the PR-HEAD input seed for CCO's canonical rerun.")
    p.add_argument("sha", nargs="?", help="git commit SHA (hex)")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        return _self_test()
    if not a.sha:
        p.error("provide a SHA, or --self-test")
    print(seed_from_sha(a.sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
