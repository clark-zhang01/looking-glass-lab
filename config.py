import os
from pathlib import Path


def _resolve_path(path_value: str, base_dir: Path) -> Path:
	candidate = Path(path_value).expanduser()
	if not candidate.is_absolute():
		candidate = base_dir / candidate
	return candidate.resolve()


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
OUTPUT_DIR = PROJECT_ROOT / "output"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

SF3D_MODEL_PATH = _resolve_path(
	os.getenv("SF3D_MODEL_PATH", str(WEIGHTS_DIR / "stable-fast-3d")),
	PROJECT_ROOT,
)

# Device Assignments
DEVICE_REMBG = os.getenv("DEVICE_REMBG", "cuda:0")
DEVICE_TRIPOSR = os.getenv("DEVICE_TRIPOSR", "cuda:0")
DEVICE_STABLEFAST3D = os.getenv("DEVICE_STABLEFAST3D", "cuda:1")

# Model IDs
TRIPOSR_MODEL_ID = "stabilityai/TripoSR"
STABLEFAST3D_MODEL_ID = "stabilityai/stable-fast-3d"

# Generation Settings
TRIPOSR_MC_RES = 128
STABLEFAST3D_MC_RES = 256

# Model Loading
LOAD_LOCAL_ONLY = True
