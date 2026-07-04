"""CPU-only tests for cross-platform host RAM detection.

Run:  python tests/test_host_memory.py
"""
import ctypes
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matmul.backend import _host_available_bytes as matmul_host_bytes
from strategy.backend import _host_available_bytes as strategy_host_bytes


def test_sysconf_path_used_when_available():
    def fake_sysconf(name):
        return 1024 if name == "SC_AVPHYS_PAGES" else 4096

    with patch("matmul.backend.os.sysconf", fake_sysconf, create=True), patch(
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

    with patch("matmul.backend.sys.platform", "win32"), patch(
        "strategy.backend.sys.platform", "win32"
    ), patch("matmul.backend.os.sysconf", side_effect=OSError("no sysconf"), create=True), patch(
        "strategy.backend.os.sysconf", side_effect=OSError("no sysconf"), create=True
    ), patch("ctypes.windll.kernel32.GlobalMemoryStatusEx", fake_gms):
        assert matmul_host_bytes() == 24 * 1024**3
        assert strategy_host_bytes() == 24 * 1024**3


def test_last_resort_fallback():
    with patch("matmul.backend.sys.platform", "linux"), patch(
        "strategy.backend.sys.platform", "linux"
    ), patch("matmul.backend.os.sysconf", side_effect=OSError("no sysconf"), create=True), patch(
        "strategy.backend.os.sysconf", side_effect=OSError("no sysconf"), create=True
    ):
        fallback = 8 * 1024**3
        assert matmul_host_bytes() == fallback
        assert strategy_host_bytes() == fallback


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
