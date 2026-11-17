from __future__ import annotations

import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str = "cuda:0") -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def make_run_id(prefix: str = "exp") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def check_disk_space(path: str | Path, min_gb: float) -> None:
    usage = shutil.disk_usage(str(path))
    free_gb = usage.free / (1024 ** 3)
    if free_gb < float(min_gb):
        raise RuntimeError(
            f"insufficient free space at {path}: {free_gb:.1f} GB "
            f"< {min_gb:.1f} GB")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True).strip()
    except Exception:
        return "unknown"


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, payload: dict) -> None:
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
