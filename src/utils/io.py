from __future__ import annotations
from pathlib import Path
import json
import joblib

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def save_joblib(obj, path: Path) -> None:
    ensure_dir(path.parent)
    joblib.dump(obj, path)

def load_joblib(path: Path):
    return joblib.load(path)

def save_fig(fig, path: Path, dpi: int = 200) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
