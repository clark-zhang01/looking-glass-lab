import gc
import logging
import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "stable-fast-3d"))
from sf3d.system import SF3D
from config import SF3D_MODEL_PATH, STABLEFAST3D_MODEL_ID

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelManager")


def _ensure_sf3d_weights(model_path: Path) -> Path:
    model_path.mkdir(parents=True, exist_ok=True)
    config_file = model_path / "config.yaml"
    weight_file = model_path / "model.safetensors"

    if config_file.exists() and weight_file.exists():
        logger.info("Using existing weights from %s", model_path)
        return model_path

    logger.info("Downloading weights for the first time to %s", model_path)
    snapshot_download(
        repo_id=STABLEFAST3D_MODEL_ID,
        local_dir=str(model_path),
        local_dir_use_symlinks=False,
    )

    if not (config_file.exists() and weight_file.exists()):
        raise FileNotFoundError(
            f"StableFast3D files were not found after download in: {model_path}"
        )

    return model_path

class ModelManager:
    """
    Manages the loading and memory cleanup of 3D generation and segmentation models.
    Designed for a multi-GPU setup with strict VRAM constraints (6GB per GPU).
    """
    def __init__(self):
        self.loaded_models = {}
        
        # Configure 4-bit quantization (NF4) for memory efficiency
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    def _check_and_free_vram(self, device_id: int, threshold_mb: float = 2048.0):
        """
        Checks the available VRAM on the target GPU. If it's below the threshold,
        triggers a memory flush.
        
        Args:
            device_id (int): The GPU ID to check.
            threshold_mb (float): The minimum required free VRAM in MB.
        """
        torch.cuda.synchronize(device_id)
        free_mem, total_mem = torch.cuda.mem_get_info(device_id)
        free_mb = free_mem / (1024**2)
        
        logger.info(f"Current free VRAM on cuda:{device_id}: {free_mb:.2f} MB")
        
        if free_mb < threshold_mb:
            logger.warning(f"Free VRAM ({free_mb:.2f} MB) is below threshold ({threshold_mb} MB). Triggering memory flush...")
            self.flush_memory()
            
            # Re-check after flush
            torch.cuda.synchronize(device_id)
            free_mem_after, _ = torch.cuda.mem_get_info(device_id)
            free_mb_after = free_mem_after / (1024**2)
            logger.info(f"Free VRAM after flush on cuda:{device_id}: {free_mb_after:.2f} MB")

    def load_model(self, model_id: str, device_id: int, model_class=AutoModelForCausalLM, local_files_only: bool = True):
        """
        Loads a model onto a specific GPU using 4-bit quantization.
        
        Args:
            model_id (str): The Hugging Face model ID.
            device_id (int): The GPU ID to load the model onto (e.g., 0 or 1).
            model_class: The Transformers model class to use for loading.
            local_files_only (bool): If True, avoids checking Hugging Face for updates.
            
        Returns:
            The loaded model instance.
        """
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. Cannot load model on GPU.")
            raise RuntimeError("CUDA is required but not available.")
            
        if device_id >= torch.cuda.device_count():
            logger.error(f"Invalid device_id: {device_id}. Only {torch.cuda.device_count()} GPUs available.")
            raise ValueError(f"Invalid device_id: {device_id}")

        device_string = f"cuda:{device_id}"
        logger.info(f"Attempting to load model '{model_id}' onto {device_string} (local_files_only={local_files_only})...")

        # VRAM Guard: Check and free memory if needed before loading
        self._check_and_free_vram(device_id)

        try:
            # Record VRAM before loading
            torch.cuda.synchronize(device_id)
            free_mem_before, _ = torch.cuda.mem_get_info(device_id)
            
            # Load the model
            # Note: We use device_map="auto" to let accelerate handle device mapping
            # We also enforce use_safetensors=True to avoid torch.load vulnerabilities
            model = model_class.from_pretrained(
                model_id,
                quantization_config=self.quantization_config,
                device_map="auto",
                use_safetensors=True,
                local_files_only=local_files_only
            )
            
            # Record VRAM after loading
            torch.cuda.synchronize(device_id)
            free_mem_after, _ = torch.cuda.mem_get_info(device_id)
            
            vram_consumed_mb = (free_mem_before - free_mem_after) / (1024**2)
            
            # get_memory_footprint might not be available on all model types
            try:
                model_footprint_mb = model.get_memory_footprint() / (1024**2)
                logger.info(f"Model memory footprint: {model_footprint_mb:.2f} MB")
            except AttributeError:
                pass
                
            logger.info(f"✅ Successfully loaded '{model_id}' on {device_string}.")
            logger.info(f"Actual VRAM consumed on {device_string}: {vram_consumed_mb:.2f} MB")
            
            # Store reference to prevent garbage collection if needed
            self.loaded_models[model_id] = {
                "model": model,
                "device": device_id
            }
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load model '{model_id}' on {device_string}: {str(e)}")
            raise

    def load_stablefast3d(self, device_id: int = 1):
        """
        Loads StableFast3D from a forced local snapshot path using the SF3D class.
        This avoids AutoModel config checks and enforces strict VRAM safety.
        """
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. Cannot load StableFast3D on GPU.")
            raise RuntimeError("CUDA is required but not available.")

        if device_id >= torch.cuda.device_count():
            logger.error(f"Invalid device_id: {device_id}. Only {torch.cuda.device_count()} GPUs available.")
            raise ValueError(f"Invalid device_id: {device_id}")

        device_string = f"cuda:{device_id}"
        sf3d_model_path_obj = _ensure_sf3d_weights(SF3D_MODEL_PATH)
        sf3d_model_path = str(sf3d_model_path_obj)
        logger.info(f"Loading StableFast3D from forced local path on {device_string}: {sf3d_model_path}")

        if not os.path.isdir(sf3d_model_path):
            raise FileNotFoundError(f"StableFast3D local snapshot path not found: {sf3d_model_path}")

        if device_id == 1:
            logger.info("Flushing GPU memory before StableFast3D load on cuda:1...")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize(device_id)

            free_mem, total_mem = torch.cuda.mem_get_info(device_id)
            free_gb = free_mem / (1024**3)
            total_gb = total_mem / (1024**3)
            logger.info(f"Available VRAM on cuda:1 before load: {free_gb:.2f} GB / {total_gb:.2f} GB")

            if free_gb < 5.0:
                raise RuntimeError(
                    f"Insufficient VRAM on cuda:1 ({free_gb:.2f} GB free). "
                    "StableFast3D load aborted; at least 5.0 GB free is required."
                )

        model = SF3D.from_pretrained(
            sf3d_model_path,
            config_name="config.yaml",
            weight_name="model.safetensors",
            local_files_only=True,
            trust_remote_code=True,
        )
        model.to(device_string)
        model.eval()

        self.loaded_models["stabilityai/stable-fast-3d"] = {
            "model": model,
            "device": device_id,
        }

        logger.info("✅ StableFast3D loaded successfully from local snapshot path.")
        return model

    def flush_memory(self):
        """
        Clears unused memory from both GPUs and runs Python garbage collection.
        """
        logger.info("Flushing memory and clearing CUDA cache...")
        
        # Run Python garbage collection
        gc.collect()
        
        # Clear CUDA cache for all available devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(f"cuda:{i}"):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    
            logger.info("✅ Memory flush complete.")
        else:
            logger.warning("CUDA not available, only Python GC was run.")

    def unload_model(self, model_id: str):
        """
        Unloads a specific model and frees its memory.
        """
        if model_id in self.loaded_models:
            logger.info(f"Unloading model '{model_id}'...")
            del self.loaded_models[model_id]["model"]
            del self.loaded_models[model_id]
            self.flush_memory()
        else:
            logger.warning(f"Model '{model_id}' is not currently loaded.")

    def test_loading(self):
        """
        Specific test case: Attempts to load 'stabilityai/TripoSR' (or fallback) 
        to cuda:0 using the 4-bit config.
        """
        logger.info("Starting specific test case: Loading TripoSR/SAM...")
        
        # Try TripoSR first, fallback to SAM if it fails (e.g., due to custom code requirements)
        primary_model = "stabilityai/TripoSR"
        fallback_model = "facebook/sam-vit-base"
        
        try:
            # TripoSR might require trust_remote_code=True, but we try standard loading first
            # Note: TripoSR is not a standard AutoModelForCausalLM, so we use AutoModel
            self.load_model(primary_model, device_id=0, model_class=AutoModel)
            self.unload_model(primary_model)
        except Exception as e:
            logger.warning(f"Failed to load {primary_model}. This might be because it requires custom code or specific pipeline classes. Error: {e}")
            logger.info(f"Falling back to {fallback_model}...")
            try:
                self.load_model(fallback_model, device_id=0, model_class=AutoModel)
                self.unload_model(fallback_model)
            except Exception as fallback_e:
                logger.error(f"Fallback model also failed: {fallback_e}")

if __name__ == "__main__":
    manager = ModelManager()
    manager.test_loading()
