"""
cco/blob.py — assemble the bound score blob (Step 9).

The canonical artifact the gate pipeline verifies. It bundles the scored latency sample + the
correctness verdict with binding/identity hashes, so a score PROVES what produced it:

  * harness_self_hash — the harness source (benchmark.py + cco/*.py): which scorer ran;
  * reference_hash    — the oracle + config for this track: a score can't be computed against a
                        weaker/edited oracle;
  * kernel_sha256     — the submitted artifact that was scored;
  * input_seed        — the (PR-HEAD-derived) seed the inputs came from;
  * gpu               — the SKU it ran on (speedups are only comparable on identical hardware).

`blob_sha256` is sha256 over all the other fields — the value that gets bound into the TDX/
attestation quote (or, in the v1 trusted-box posture, the integrity hash that ties the score to
its evidence). The harness makes NO keep/revert decision; the blob is just evidence the gate
pipeline checks before comparing challenger-vs-champion.

Usage: uv run --no-project python cco/blob.py --self-test
"""

from __future__ import annotations

import hashlib
import json
import os


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_files(root: str, paths: list[str]) -> str:
    """Stable aggregate hash of a file set: sha256 of sorted 'relpath:filehash' lines.

    Uses repo-relative paths as labels so same-named files in different dirs (e.g.
    references/matmul.py vs kernel_configs/matmul.py) don't collide.
    """
    parts = []
    for p in sorted(paths):
        rel = os.path.relpath(p, root).replace("\\", "/")
        digest = _sha256_file(p) if os.path.isfile(p) else "MISSING"
        parts.append(f"{rel}:{digest}")
    return _sha256_text("\n".join(parts))


def harness_self_hash(repo_root: str, harness_path: str) -> str:
    """Hash the harness source: the harness file + every cco/*.py module."""
    files = [harness_path]
    cco_dir = os.path.join(repo_root, "cco")
    if os.path.isdir(cco_dir):
        files += [os.path.join(cco_dir, f) for f in os.listdir(cco_dir) if f.endswith(".py")]
    return _hash_files(repo_root, files)


def reference_hash(repo_root: str, kernel_type: str) -> str:
    """Hash the locked oracle + config for this track."""
    files = [
        os.path.join(repo_root, "references", f"{kernel_type}.py"),
        os.path.join(repo_root, "references", "__init__.py"),
        os.path.join(repo_root, "kernel_configs", f"{kernel_type}.py"),
        os.path.join(repo_root, "kernel_configs", f"{kernel_type}.toml"),
        os.path.join(repo_root, "kernel_configs", "_utils.py"),
        os.path.join(repo_root, "kernel_configs", "__init__.py"),
    ]
    return _hash_files(repo_root, files)


def build_score_blob(*, competition: str, version: int, kernel_type: str, seed: int,
                     correctness: dict, scored, gpu: dict,
                     repo_root: str, harness_path: str, kernel_path) -> dict:
    """Assemble the bound score blob and stamp its blob_sha256 (over all other fields)."""
    blob = {
        "competition": competition,
        "version": version,
        "kernel_type": kernel_type,
        "input_seed": seed,
        "correctness": correctness,
        "scored": scored,
        "gpu": gpu,
        "kernel_sha256": (_sha256_file(kernel_path)
                          if kernel_path and os.path.isfile(kernel_path) else None),
        "harness_self_hash": harness_self_hash(repo_root, harness_path),
        "reference_hash": reference_hash(repo_root, kernel_type),
    }
    blob["blob_sha256"] = _sha256_text(json.dumps(blob, sort_keys=True, separators=(",", ":")))
    return blob


# --------------------------------------------------------------------------------------
# Self-test (pure Python; builds a blob against a temp repo)
# --------------------------------------------------------------------------------------

def _self_test() -> int:
    import shutil
    import tempfile

    failures = 0

    def check(cond, label):
        nonlocal failures
        print(("ok   " if cond else "FAIL ") + label)
        if not cond:
            failures += 1

    tmp = tempfile.mkdtemp(prefix="cco_blob_")
    try:
        def w(rel, content):
            p = os.path.join(tmp, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            return p

        harness = w("benchmark.py", "# harness v1\n")
        w("cco/guard_kernel.py", "# guard\n")
        w("references/__init__.py", "")
        w("references/matmul.py", "def matmul_ref(): pass\n")
        w("kernel_configs/__init__.py", "")
        w("kernel_configs/_utils.py", "")
        w("kernel_configs/matmul.py", "# config\n")
        w("kernel_configs/matmul.toml", "x=1\n")
        kpath = w("kernel.py", "def kernel_fn(): pass\n")

        common = dict(competition="CCO", version=1, kernel_type="matmul", seed=123,
                      correctness={"overall": "PASS"},
                      scored={"median_us": 100.0, "latencies_us": [99.0, 101.0]},
                      gpu={"name": "RTX 5070 Ti", "compute_capability": "12.0"},
                      repo_root=tmp, harness_path=harness, kernel_path=kpath)

        b1 = build_score_blob(**common)
        check(all(len(b1[k]) == 64 for k in ("blob_sha256", "kernel_sha256",
                                             "harness_self_hash", "reference_hash")),
              "blob has 64-hex blob/kernel/harness/reference hashes")

        b2 = build_score_blob(**common)
        check(b1["blob_sha256"] == b2["blob_sha256"], "blob_sha256 deterministic for identical inputs")

        c3 = dict(common)
        c3["scored"] = {"median_us": 90.0, "latencies_us": [89.0, 91.0]}
        b3 = build_score_blob(**c3)
        check(b3["blob_sha256"] != b1["blob_sha256"], "different score -> different blob_sha256")

        w("kernel.py", "def kernel_fn(): return 1  # changed\n")
        b4 = build_score_blob(**common)
        check(b4["kernel_sha256"] != b1["kernel_sha256"], "edited kernel -> different kernel_sha256")
        check(b4["blob_sha256"] != b1["blob_sha256"], "edited kernel -> different blob_sha256")

        w("references/matmul.py", "def matmul_ref(): return 0  # tampered oracle\n")
        b5 = build_score_blob(**common)
        check(b5["reference_hash"] != b1["reference_hash"], "tampered oracle -> different reference_hash")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("-" * 60)
    print("SELF-TEST PASSED" if not failures else f"SELF-TEST FAILED: {failures} case(s)")
    return 1 if failures else 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Bound score-blob builder (CCO).")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        return _self_test()
    p.error("pass --self-test (or import build_score_blob)")


if __name__ == "__main__":
    raise SystemExit(main())
