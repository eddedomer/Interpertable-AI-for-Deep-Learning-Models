@echo off
setlocal enabledelayedexpansion

REM Usage:
REM   scripts\setup_windows.bat cpu
REM   scripts\setup_windows.bat cu126

set TORCH=%1
if "%TORCH%"=="" set TORCH=cpu

REM Move to repo root (parent of scripts)
cd /d "%~dp0\.."

echo == ARI3205 Windows setup (.bat) ==
echo Torch mode: %TORCH%

REM 1) Create venv
if not exist .venv (
  echo Creating venv...
  python -m venv .venv
)

REM 2) Activate venv
call .venv\Scripts\activate.bat

REM 3) Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel

REM 4) Install requirements
if not exist requirements\windows.txt (
  echo ERROR: requirements\windows.txt not found
  exit /b 1
)

echo Installing requirements from requirements\windows.txt ...
pip install -r requirements\windows.txt

REM 5) Optional: reinstall torch from PyTorch index for CUDA builds
if /I not "%TORCH%"=="cpu" (
  echo Reinstalling PyTorch for %TORCH% ...
  pip uninstall -y torch torchvision torchaudio

  set INDEXURL=https://download.pytorch.org/whl/%TORCH%
  pip install torch torchvision torchaudio --index-url %INDEXURL%
)

REM 6) Register kernel
python -m ipykernel install --user --name ari3205 --display-name "ARI3205 (.venv)"

echo.
echo Done ✅
echo Next: jupyter lab
endlocal
