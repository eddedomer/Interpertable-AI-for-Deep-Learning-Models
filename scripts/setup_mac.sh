#!/usr/bin/env bash
set -euo pipefail

# ---------------- Defaults ----------------
VENV_DIR=".venv"
KERNEL_NAME="ari3205"
KERNEL_DISPLAY="ARI3205 (.venv)"
PYTHON_BIN="python3"
REQ_FILE="requirements/mac.txt"

# ---------------- Args ----------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV_DIR="$2"; shift 2 ;;
    --kernel-name) KERNEL_NAME="$2"; shift 2 ;;
    --kernel-display) KERNEL_DISPLAY="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --req) REQ_FILE="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: ./scripts/setup_mac.sh [options]

Options:
  --venv <dir>            Virtual env directory (default: .venv)
  --kernel-name <name>    Jupyter kernel name (default: ari3205)
  --kernel-display <str>  Jupyter display name (default: "ARI3205 (.venv)")
  --python <bin>          Python executable (default: python3)
  --req <file>            Requirements file (default: requirements/mac.txt)
EOF
      exit 0
      ;;
    *) echo "Unknown arg: $1" && exit 1 ;;
  esac
done

# ---------------- Repo root ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "== ARI3205 macOS setup =="
echo "Repo: $REPO_ROOT"
echo "Python: $PYTHON_BIN"

# ---------------- Python check ----------------
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: $PYTHON_BIN not found"; exit 1; }
"$PYTHON_BIN" --version

# ---------------- Create venv ----------------
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# ---------------- Activate venv ----------------
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel

# ---------------- Install requirements ----------------
if [[ ! -f "$REQ_FILE" ]]; then
  echo "ERROR: Requirements file not found: $REQ_FILE"
  exit 1
fi

echo "Installing requirements from $REQ_FILE ..."
pip install -r "$REQ_FILE"

# ---------------- TensorFlow Metal (only on Apple Silicon) ----------------
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  echo "Installing tensorflow-metal (Apple Silicon acceleration)..."
  # If it fails, keep going (CPU TF still works)
  pip install tensorflow-metal || echo "WARNING: tensorflow-metal install failed; continuing with CPU TensorFlow."
fi

# ---------------- Jupyter kernel ----------------
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"

echo ""
echo "Done ✅"
echo "Next:"
echo "  source $VENV_DIR/bin/activate"
echo "  jupyter lab"
echo "  Select kernel: $KERNEL_DISPLAY"
