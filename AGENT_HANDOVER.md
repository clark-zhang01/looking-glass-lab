# AGENT HANDOVER

## Project Overview
This project generates **3D `.glb` assets** from **2D input images** for display on a **Looking Glass Portrait**.  
The backend serves generation APIs and static assets; the frontend provides a Three.js/WebXR viewing interface.

## Tech Stack
- **Backend**: FastAPI
- **3D Models**: TripoSR + StableFast3D
- **Frontend**: Three.js
- **Holographic Runtime**: Looking Glass WebXR bridge (`webxr.js`)

## Hardware Config (Target Machine)
- **GPU**: Dual RTX 4090
- Expected multi-GPU mapping:
  - `cuda:0` for lighter/2D-side workloads (e.g., TripoSR + rembg)
  - `cuda:1` for StableFast3D
- Device routing is controlled via environment variables in `config.py`:
  - `DEVICE_REMBG`
  - `DEVICE_TRIPOSR`
  - `DEVICE_STABLEFAST3D`

Example GPU assignment at runtime:
```bash
DEVICE_REMBG=cuda:0 DEVICE_TRIPOSR=cuda:0 DEVICE_STABLEFAST3D=cuda:1 ./reboot_lab.sh
```

## Critical Paths (Dynamic Model Pathing)
Path management is dynamic and rooted at project directory:
- `PROJECT_ROOT` is defined in `config.py`
- `STATIC_DIR`, `OUTPUT_DIR`, `WEIGHTS_DIR` are derived from `PROJECT_ROOT`
- StableFast3D path uses:
  - `SF3D_MODEL_PATH` env var if set
  - otherwise defaults to `PROJECT_ROOT/weights/stable-fast-3d`

StableFast3D loading logic (in `core/generator.py` and `core/loader.py`) now:
1. Checks if `config.yaml` and `model.safetensors` exist in `SF3D_MODEL_PATH`
2. If present: logs **"Using existing weights"**
3. If missing: runs `huggingface_hub.snapshot_download(..., local_dir_use_symlinks=False)` and logs **"Downloading weights for the first time"**

## Deployment Steps (New Machine)
From project root:

```bash
# 1) Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) (Optional) set explicit SF3D model path
export SF3D_MODEL_PATH="$PWD/weights/stable-fast-3d"

# 4) (Optional) set GPU assignment for dual-4090
export DEVICE_REMBG=cuda:0
export DEVICE_TRIPOSR=cuda:0
export DEVICE_STABLEFAST3D=cuda:1

# 5) Restart server via helper script
./reboot_lab.sh
```

## WebXR Setup Note
During setup, ensure Looking Glass WebXR library exists at:
- `static/libs/webxr.js`

If missing, download with:
```bash
mkdir -p static/libs
curl -L -o static/libs/webxr.js https://unpkg.com/@lookingglass/webxr@0.6.0/dist/webxr.js
```

`index.html` should load `webxr.js` before `static/js/main.js`.
