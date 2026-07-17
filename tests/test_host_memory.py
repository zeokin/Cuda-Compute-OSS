"""CPU-only tests for cross-platform host RAM detection.

_host_available_bytes() consults Linux MemAvailable first, so every test below
that exercises a *fallback* (sysconf / win32 / last-resort) stubs
_linux_mem_available -> None; otherwise the real /proc/meminfo on the CI box
would answer first and the fallback under test would never run.

Run:  python tests/test_host_memory.py
"""
import ctypes
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matmul.backend import _host_available_bytes as matmul_host_bytes
from strategy.backend import _host_available_bytes as strategy_host_bytes
from strategy.cpu_backend import CPUBackend


def _no_meminfo():
    """Patch both copies' MemAvailable probe to 'unreadable' so the sysconf /
    win32 / last-resort fallbacks are the code path actually under test."""
    return (
        patch("matmul.backend._linux_mem_available", lambda: None),
        patch("strategy.backend._linux_mem_available", lambda: None),
    )


def test_sysconf_path_used_when_available():
    def fake_sysconf(name):
        return 1024 if name == "SC_AVPHYS_PAGES" else 4096

    mm, st = _no_meminfo()
    with mm, st, patch("matmul.backend.os.sysconf", fake_sysconf, create=True), patch(
        "strategy.backend.os.sysconf", fake_sysconf, create=True
    ):
        assert matmul_host_bytes() == 1024 * 4096
        assert strategy_host_bytes() == 1024 * 4096


def test_windows_global_memory_status_ex():
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

    def fake_gms(buf):
        stat = ctypes.cast(buf, ctypes.POINTER(_MEMORYSTATUSEX)).contents
        stat.ullAvailPhys = 24 * 1024**3
        return 1

    # ctypes.windll only exists on actual Windows -- patch(..., create=True)
    # on the whole attribute (not a nested path under it) so this test can
    # run and mean something on the Linux/macOS CI box that actually runs it.
    fake_windll = MagicMock()
    fake_windll.kernel32.GlobalMemoryStatusEx = fake_gms

    mm, st = _no_meminfo()
    with mm, st, patch("matmul.backend.sys.platform", "win32"), patch(
        "strategy.backend.sys.platform", "win32"
    ), patch("matmul.backend.os.sysconf", side_effect=OSError("no sysconf"), create=True), patch(
        "strategy.backend.os.sysconf", side_effect=OSError("no sysconf"), create=True
    ), patch("ctypes.windll", fake_windll, create=True):
        assert matmul_host_bytes() == 24 * 1024**3
        assert strategy_host_bytes() == 24 * 1024**3


def test_cpu_backend_uses_shared_sysconf_path():
    # CPUBackend.host_available_bytes must delegate to the same cross-platform
    # helper as the GPU backends, not a private, incomplete copy.
    def fake_sysconf(name):
        return 2048 if name == "SC_AVPHYS_PAGES" else 4096

    mm, st = _no_meminfo()
    with mm, st, patch("strategy.backend.os.sysconf", fake_sysconf, create=True):
        backend = CPUBackend(verbose=False)
        assert backend.host_available_bytes() == 2048 * 4096
        assert backend.free_compute_bytes() == 2048 * 4096


def test_cpu_backend_uses_windows_global_memory_status_ex():
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

    def fake_gms(buf):
        stat = ctypes.cast(buf, ctypes.POINTER(_MEMORYSTATUSEX)).contents
        stat.ullAvailPhys = 24 * 1024**3
        return 1

    fake_windll = MagicMock()
    fake_windll.kernel32.GlobalMemoryStatusEx = fake_gms

    mm, st = _no_meminfo()
    with mm, st, patch("strategy.backend.sys.platform", "win32"), patch(
        "strategy.backend.os.sysconf", side_effect=OSError("no sysconf"), create=True
    ), patch("ctypes.windll", fake_windll, create=True):
        backend = CPUBackend(verbose=False)
        # Before the fix this returned a hardcoded 2 GiB, ignoring real RAM.
        assert backend.host_available_bytes() == 24 * 1024**3


def test_last_resort_fallback():
    mm, st = _no_meminfo()
    with mm, st, patch("matmul.backend.sys.platform", "linux"), patch(
        "strategy.backend.sys.platform", "linux"
    ), patch("matmul.backend.os.sysconf", side_effect=OSError("no sysconf"), create=True), patch(
        "strategy.backend.os.sysconf", side_effect=OSError("no sysconf"), create=True
    ):
        fallback = 8 * 1024**3
        assert matmul_host_bytes() == fallback
        assert strategy_host_bytes() == fallback


def test_mem_available_preferred_over_sysconf_free():
    # The whole point: SC_AVPHYS_PAGES is MemFree (excludes reclaimable page
    # cache) and under-reports. MemAvailable must win when both are readable.
    def fake_sysconf(name):
        return 1 if name == "SC_AVPHYS_PAGES" else 4096       # 4 KiB "free"

    with patch("matmul.backend._linux_mem_available", lambda: 32 * 1024**3), patch(
        "strategy.backend._linux_mem_available", lambda: 32 * 1024**3
    ), patch("matmul.backend.os.sysconf", fake_sysconf, create=True), patch(
        "strategy.backend.os.sysconf", fake_sysconf, create=True
    ):
        assert matmul_host_bytes() == 32 * 1024**3
        assert strategy_host_bytes() == 32 * 1024**3


def test_falls_back_to_sysconf_when_meminfo_unreadable():
    # Non-Linux (or a container without /proc): the probe returns None and the
    # existing sysconf path must still be used, not the last-resort constant.
    def fake_sysconf(name):
        return 1024 if name == "SC_AVPHYS_PAGES" else 4096

    mm, st = _no_meminfo()
    with mm, st, patch("matmul.backend.os.sysconf", fake_sysconf, create=True), patch(
        "strategy.backend.os.sysconf", fake_sysconf, create=True
    ):
        assert matmul_host_bytes() == 1024 * 4096
        assert strategy_host_bytes() == 1024 * 4096


def test_mem_available_parses_real_proc_meminfo():
    # On this Linux box the probe must return a sane, positive value that
    # matches /proc/meminfo's own MemAvailable line.
    import re as _re

    from matmul.backend import _linux_mem_available as mm_probe
    from strategy.backend import _linux_mem_available as st_probe

    try:
        text = open("/proc/meminfo").read()
    except OSError:
        return                                   # not Linux: nothing to check
    match = _re.search(r"^MemAvailable:\s+(\d+) kB", text, _re.M)
    if match is None:
        return                                   # kernel too old for MemAvailable
    expected = int(match.group(1)) * 1024
    for probe in (mm_probe, st_probe):
        got = probe()
        assert got is not None and got > 0
        # Memory moves between reads; allow a wide band but pin the unit/scale.
        assert 0.5 * expected <= got <= 2.0 * expected


def test_mem_available_probe_survives_a_broken_proc(tmp_path):
    # A garbage/absent /proc/meminfo must yield None (-> fall through), never raise.
    import builtins

    def boom(*a, **k):
        raise OSError("no /proc")

    with patch.object(builtins, "open", boom):
        from matmul.backend import _linux_mem_available as probe
        assert probe() is None


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
