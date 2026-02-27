# Looking Glass 3D Lab

Generate holographic-ready `.glb` assets from 2D images for **Looking Glass Portrait**.

## Overview
This project provides:
- A **FastAPI backend** for 3D asset generation
- A web UI using **Three.js + WebXR** for preview and Looking Glass output
- Support for **TripoSR** and **StableFast3D** pipelines

## Important Repository Note
To keep the repository lightweight, large model source folders are **not included**:
- `TripoSR/`
- `stable-fast-3d/`

You must provide these folders locally (same level as `app.py`) before running the app.

## Requirements
- Python 3.12+
- NVIDIA GPU + drivers (recommended)
- Git

Install Python dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Model Pathing
StableFast3D weights path is dynamic:
- Uses `SF3D_MODEL_PATH` if set
- Otherwise defaults to `weights/stable-fast-3d` under project root

Example:
```bash
export SF3D_MODEL_PATH="$PWD/weights/stable-fast-3d"
```

## WebXR Setup
Download Looking Glass WebXR bridge file:
```bash
mkdir -p static/libs
curl -L -o static/libs/webxr.js https://unpkg.com/@lookingglass/webxr@0.6.0/dist/webxr.js
```

## Run
Use the restart helper:
```bash
./reboot_lab.sh
```

Then open:
- `http://127.0.0.1:8000`

## Optional GPU Assignment (multi-GPU)
```bash
DEVICE_REMBG=cuda:0 DEVICE_TRIPOSR=cuda:0 DEVICE_STABLEFAST3D=cuda:1 ./reboot_lab.sh
```

## API Endpoints
- `GET /health` — GPU/status info
- `POST /generate` — generate `.glb` from upload or sample
