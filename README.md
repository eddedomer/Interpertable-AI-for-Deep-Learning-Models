# Interpertable-AI-for-Deep-Learning-Models

## Setup

#### Mac OS (Intel and Arm safe)
```bash
./scripts/setup_mac.sh
```
#### Windows PowerShell:

```powershell 
cd Interpertable-AI-for-Deep-Learning-Models
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

```powershell
_.\scripts\setup_windows.ps1 -Torch cpu_

# GPU if you have NVIDIA + CUDA-capable setup:

_.\scripts\setup_windows.ps1 -Torch cu126_

```


#### Windows CMD:
```bat 
scripts\setup_windows.bat cpu
REM or 
scripts\setup_windows.bat cu126
```