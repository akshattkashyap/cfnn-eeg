"""Atomic JSON reports and runtime environment metadata."""
from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .models import tf


def write_json(path, value):
    """Replace atomically so an interrupted run does not leave a half-written report."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def environment_info():
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "tensorflow", "keras", "joblib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages,
            "gpus": [d.name for d in tf.config.list_physical_devices("GPU")] if tf else []}
