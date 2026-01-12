from pathlib import Path

# repo_root/src/config.py -> repo_root
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"

def ensure_dirs() -> None:
    for p in [RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, OUTPUTS_DIR, REPORTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def artifact_dir(name: str) -> Path:
    d = ARTIFACTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d

def output_dir(*parts: str) -> Path:
    d = OUTPUTS_DIR.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
