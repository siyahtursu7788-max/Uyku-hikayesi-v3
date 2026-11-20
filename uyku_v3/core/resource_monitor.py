"""UYKU_HIKAYESI_V3_FINAL_PRODUCTION — T4 GPU VRAM, RAM ve disk izleme yöneticisi."""

from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Any, Dict

try:
    import psutil as _psutil
except Exception:
    _psutil = None

try:
    import torch as _torch
except Exception:
    _torch = None


def get_ram_metrics() -> Dict[str, float]:
    if _psutil is not None:
        vm = _psutil.virtual_memory()
        return {
            "used_mb": round(vm.used / (1024 * 1024), 1),
            "total_mb": round(vm.total / (1024 * 1024), 1),
            "percent": float(vm.percent),
        }
    info: Dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                info[parts[0].rstrip(":")] = int(parts[1])
        total_kb = info.get("MemTotal", 0)
        avail_kb = info.get("MemAvailable", info.get("MemFree", 0))
        used_kb = max(0, total_kb - avail_kb)
        return {
            "used_mb": round(used_kb / 1024.0, 1),
            "total_mb": round(total_kb / 1024.0, 1),
            "percent": round((used_kb / total_kb * 100.0) if total_kb else 0.0, 1),
        }
    except Exception:
        return {"used_mb": 0.0, "total_mb": 0.0, "percent": 0.0}


def get_gpu_metrics() -> Dict[str, Any]:
    if _torch is None:
        return {"cuda_available": False, "device_name": "CPU_ONLY", "allocated_mb": 0.0, "peak_mb": 0.0}
    has_cuda = bool(getattr(_torch.cuda, "is_available")())
    if not has_cuda:
        return {"cuda_available": False, "device_name": "CPU_ONLY", "allocated_mb": 0.0, "peak_mb": 0.0}
    return {
        "cuda_available": True,
        "device_name": _torch.cuda.get_device_name(0),
        "allocated_mb": round(_torch.cuda.memory_allocated(0) / (1024 * 1024), 1),
        "peak_mb": round(_torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1),
    }


def get_disk_free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return round(shutil.disk_usage(str(p)).free / (1024 ** 3), 2)


def release_gpu_vram() -> None:
    gc.collect()
    if _torch is not None and bool(getattr(_torch.cuda, "is_available")()):
        try:
            _torch.cuda.empty_cache()
            _torch.cuda.ipc_collect()
        except Exception:
            pass


def snapshot_system_resources(work_dir: Path) -> Dict[str, Any]:
    return {
        "gpu": get_gpu_metrics(),
        "ram": get_ram_metrics(),
        "disk_free_gb": get_disk_free_gb(work_dir),
    }
