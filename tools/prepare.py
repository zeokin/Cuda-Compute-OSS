"""Environment preparation and validation for cuda-evolve."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "workspace" / "results.tsv"
MEMORY_FILE = ROOT / "workspace" / "MEMORY.md"


def check_python():
    v = sys.version_info
    print(f"[✓] Python {v.major}.{v.minor}.{v.micro}")
    if v < (3, 10):
        print("[✗] Python >= 3.10 required")
        sys.exit(1)


def check_cuda():
    try:
        import torch

        if not torch.cuda.is_available():
            print("[✗] CUDA is not available. A CUDA-capable GPU is required.")
            sys.exit(1)

        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        print(f"[✓] CUDA available: {device_name} (SM {capability[0]}{capability[1]}, {vram_mb:.0f} MB)")
    except ImportError:
        print("[✗] PyTorch not installed. Run: uv sync")
        sys.exit(1)


def check_triton():
    try:
        import triton

        print(f"[✓] Triton {triton.__version__}")
    except ImportError:
        print("[!] Triton not installed — Triton kernels will not be available")


def check_tools():
    for tool, desc in [("nvcc", "CUDA Compiler"), ("ncu", "Nsight Compute"), ("nsys", "Nsight Systems")]:
        path = shutil.which(tool)
        if path:
            print(f"[✓] {desc}: {path}")
            if tool == "nvidia-smi":
                print("nvidia_smi=ok")
        else:
            print(f"[!] {desc} ({tool}) not found in PATH — profiling features may be limited")
            if tool == "nvidia-smi":
                print("nvidia_smi=missing")


def check_git():
    try:
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT)
        if result.returncode == 0:
            print("[✓] Git repository OK")
        else:
            print("[✗] Not a git repository")
            sys.exit(1)
    except FileNotFoundError:
        print("[✗] git not found")
        sys.exit(1)


RESULTS_HEADER = (
    "experiment_id\thypothesis\tcorrectness\ttime_ms\tthroughput\tpeak_vram_mb\tkept"
    "\tpct_peak_compute\tpct_peak_bandwidth\tbottleneck\tgit_sha\tparent_experiment_id"
    "\tncu_top_stall\tncu_occupancy\tncu_l1_hit_rate\tncu_l2_hit_rate"
    "\n"
)


def init_results():
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(RESULTS_HEADER)
        print(f"[✓] Created {RESULTS_FILE.name}")
    else:
        lines = RESULTS_FILE.read_text().strip().split("\n")
        header = lines[0] if lines else ""
        if "git_sha" not in header:
            print(f"[!] Migrating {RESULTS_FILE.name} to extended schema")
            old_rows = lines[1:] if len(lines) > 1 else []
            old_col_count = len(header.split("\t")) if header else 7
            new_col_count = len(RESULTS_HEADER.strip().split("\t"))
            migrated = [RESULTS_HEADER.strip()]
            for row in old_rows:
                cols = row.split("\t")
                cols.extend([""] * (new_col_count - len(cols)))
                migrated.append("\t".join(cols[:new_col_count]))
            RESULTS_FILE.write_text("\n".join(migrated) + "\n")
            print(
                f"[✓] Migrated {RESULTS_FILE.name} "
                f"({len(old_rows)} rows, {old_col_count} -> {new_col_count} columns)"
            )
        else:
            print(f"[✓] {RESULTS_FILE.name} exists ({len(lines) - 1} experiments recorded)")


def init_memory():
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(
            "# Optimization Log\n\n"
            "This file records the history of optimization experiments.\n\n---\n\n"
            "<!-- New entries should be added below this line, in reverse chronological order. -->\n"
        )
        print(f"[✓] Created {MEMORY_FILE.name}")
    else:
        print(f"[✓] {MEMORY_FILE.name} exists")


def check_kernel_files():
    kernel_py = ROOT / "kernel.py"
    references_dir = ROOT / "references"

    if kernel_py.exists():
        print(f"[✓] {kernel_py.name} exists")
    else:
        print(f"[!] {kernel_py.name} not found — create it or copy a kernel from kernels/")

    if references_dir.exists() and (references_dir / "__init__.py").exists():
        print("[✓] references/ package exists")
    else:
        print("[!] references/ package not found — create it before running experiments")


def main():
    print("=" * 60)
    print("  cuda-evolve Environment Check")
    print("=" * 60)
    print()

    check_python()
    check_cuda()
    check_triton()
    print()
    check_tools()
    print()
    check_git()
    init_results()
    init_memory()
    print()
    check_kernel_files()

    print()
    print("=" * 60)
    print("  Environment ready. Read program.md to begin.")
    print("=" * 60)


if __name__ == "__main__":
    main()
