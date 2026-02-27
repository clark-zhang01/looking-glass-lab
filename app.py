import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
import logging
import torch
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from core.loader import ModelManager
from core.generator import TripoSRGenerator, StableFast3DGenerator
from config import STATIC_DIR, OUTPUT_DIR, TRIPOSR_MC_RES, STABLEFAST3D_MC_RES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FastAPI_App")

# Define directories
UPLOADS_DIR = OUTPUT_DIR / "uploads"
ASSETS_DIR = OUTPUT_DIR / "assets"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Global instances
model_manager = None
generators = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Initializes the ModelManager on startup and flushes memory on shutdown.
    """
    global model_manager, generators
    logger.info("Starting up FastAPI application...")
    
    # Initialize the ModelManager as a global singleton
    model_manager = ModelManager()
    logger.info("ModelManager initialized.")
    
    # Initialize the Generators
    generators['triposr'] = TripoSRGenerator()
    generators['stablefast3d'] = StableFast3DGenerator()
    
    # Pre-load the default model during startup to avoid delay on first request
    try:
        generators['triposr'].load_model()
    except Exception as e:
        logger.error(f"Failed to pre-load TripoSR model: {e}")
    
    yield # Application runs here
    
    logger.info("Shutting down FastAPI application...")
    for gen in generators.values():
        gen.unload_model()
    if model_manager:
        logger.info("Flushing memory before shutdown...")
        model_manager.flush_memory()
    logger.info("Shutdown complete.")

# Initialize FastAPI app
app = FastAPI(
    title="Looking Glass 3D Generation API",
    description="API for serving 3D assets generated via SAM 3D / TripoSR on dual GTX 1060 GPUs.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for remote access (e.g., Surface Book 2)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for asset serving
# A file at output/assets/test.glb will be accessible at /assets/test.glb
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def read_root():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/health")
async def health_check():
    """
    Health endpoint that returns the real-time VRAM status for both GPUs.
    """
    if not torch.cuda.is_available():
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "CUDA is not available."}
        )
        
    gpu_status = []
    
    def get_gpu_info(device_id):
        free_mem, total_mem = torch.cuda.mem_get_info(device_id)
        name = torch.cuda.get_device_name(device_id)
        return free_mem, total_mem, name

    for i in range(torch.cuda.device_count()):
        try:
            # Run PyTorch CUDA calls in a separate thread to avoid blocking the event loop
            free_mem, total_mem, name = await asyncio.wait_for(
                asyncio.to_thread(get_gpu_info, i),
                timeout=2.0
            )
            free_mb = free_mem / (1024**2)
            total_mb = total_mem / (1024**2)
            used_mb = total_mb - free_mb
            
            gpu_status.append({
                "device_id": i,
                "name": name,
                "total_vram_mb": round(total_mb, 2),
                "used_vram_mb": round(used_mb, 2),
                "free_vram_mb": round(free_mb, 2)
            })
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting info for GPU {i}")
            gpu_status.append({
                "device_id": i,
                "error": "Timeout getting GPU info"
            })
        except Exception as e:
            logger.error(f"Failed to get info for GPU {i}: {e}")
            gpu_status.append({
                "device_id": i,
                "error": str(e)
            })
            
    return {
        "status": "ok",
        "gpu_count": torch.cuda.device_count(),
        "gpus": gpu_status
    }

@app.post("/generate")
async def generate_3d_model(
    file: UploadFile = File(None),
    sample_path: str = Form(None),
    engine_type: str = Form("triposr")
):
    """
    Endpoint to accept an image upload or a sample path and trigger 3D generation.
    Returns the URL to the generated .glb asset.
    """
    if not file and not sample_path:
        raise HTTPException(status_code=400, detail="No file uploaded or sample selected.")
        
    if engine_type not in generators:
        raise HTTPException(status_code=400, detail=f"Invalid engine type: {engine_type}")
        
    try:
        if file and file.filename:
            # Save the uploaded image
            file_path = UPLOADS_DIR / Path(file.filename).name
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
                
            logger.info(f"Saved uploaded image to {file_path}")
            original_filename = file.filename
        elif sample_path:
            # Use the sample image
            file_path = STATIC_DIR / "samples" / Path(sample_path).name
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"Sample {sample_path} not found.")
            logger.info(f"Using sample image {file_path}")
            original_filename = sample_path
        else:
            raise HTTPException(status_code=400, detail="Invalid input.")
        
        # Generate unique filename for the output asset
        base_name = os.path.splitext(original_filename)[0]
        asset_filename = f"{base_name}_{engine_type}_3d.glb"
        asset_path = ASSETS_DIR / asset_filename
        
        # Run the generation pipeline
        logger.info(f"Starting 3D generation for {original_filename} using {engine_type}...")
        
        generator = generators[engine_type]
        mc_res = TRIPOSR_MC_RES if engine_type == "triposr" else STABLEFAST3D_MC_RES
        
        await asyncio.to_thread(
            generator.generate,
            image_path=str(file_path),
            output_path=str(asset_path),
            mc_resolution=mc_res
        )
        
        asset_url = f"/assets/{asset_filename}"
                
        return {
            "status": "success",
            "message": "3D generation completed successfully.",
            "original_filename": original_filename,
            "asset_url": asset_url,
            "engine": engine_type
        }
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Run the application using uvicorn
    # Host 0.0.0.0 allows external connections
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
