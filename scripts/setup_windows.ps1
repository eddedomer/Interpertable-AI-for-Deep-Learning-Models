Param(
  [ValidateSet("cpu","cu118","cu121","cu124","cu126","cu128")]
  [string]$Torch = "cpu",

  [string]$VenvDir = ".venv",
  [string]$KernelName = "ari3205",
  [string]$KernelDisplay = "ARI3205 (.venv)",
  [string]$ReqFile = "requirements\windows.txt"
)

$ErrorActionPreference = "Stop"

# Resolve repo root = parent of scripts/
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

Write-Host "== ARI3205 Windows setup =="
python --version

# 1) Create venv
if (!(Test-Path $VenvDir)) {
  Write-Host "Creating venv in $VenvDir ..."
  python -m venv $VenvDir
}

# 2) Activate venv
Write-Host "Activating venv..."
& "$VenvDir\Scripts\Activate.ps1"

# 3) Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel

# 4) Install requirements
if (!(Test-Path $ReqFile)) {
  throw "Requirements file not found: $ReqFile"
}
Write-Host "Installing requirements from $ReqFile ..."
pip install -r $ReqFile

# 5) Optional: reinstall torch with specific CUDA/CPU index
# Windows.txt already installs CPU torch via PyPI.
# If user requested cuXXX, reinstall from PyTorch index-url.
if ($Torch -ne "cpu") {
  Write-Host "Reinstalling PyTorch for $Torch ..."
  pip uninstall -y torch torchvision torchaudio | Out-Null

  $IndexUrl = "https://download.pytorch.org/whl/$Torch"
  pip install torch torchvision torchaudio --index-url $IndexUrl
}

# 6) Register Jupyter kernel
Write-Host "Registering Jupyter kernel..."
python -m ipykernel install --user --name $KernelName --display-name $KernelDisplay

Write-Host ""
Write-Host "Done ✅"
Write-Host "Next:"
Write-Host "  1) jupyter lab"
Write-Host "  2) Select kernel: $KernelDisplay"
