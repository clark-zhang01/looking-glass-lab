import os
import gc
import logging
from contextlib import nullcontext
from pathlib import Path
import torch
import numpy as np
import trimesh
from PIL import Image
import rembg
from huggingface_hub import snapshot_download

# Add TripoSR to path so we can import it
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "TripoSR"))
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

# Add StableFast3D to path
sys.path.append(str(PROJECT_ROOT / "stable-fast-3d"))
from sf3d.system import SF3D
from sf3d.utils import (
    create_intrinsic_from_fov_deg as sf3d_create_intrinsic_from_fov_deg,
    default_cond_c2w as sf3d_default_cond_c2w,
)
# We will use tsr.utils for background removal as they are compatible

from config import (
    DEVICE_REMBG,
    DEVICE_TRIPOSR,
    DEVICE_STABLEFAST3D,
    TRIPOSR_MODEL_ID,
    STABLEFAST3D_MODEL_ID,
    LOAD_LOCAL_ONLY,
    SF3D_MODEL_PATH,
)

logger = logging.getLogger("3DGenerator")

class BaseGenerator:
    """
    Base class for 3D generation engines.
    """
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model = None
        self.rembg_session = None

    def _resolve_supported_device(self, requested_device: str) -> str:
        if not requested_device.startswith("cuda"):
            return requested_device

        if not torch.cuda.is_available():
            logger.warning(
                "Requested %s but CUDA is unavailable. Falling back to CPU.",
                requested_device,
            )
            return "cpu"

        try:
            device_id = int(requested_device.split(":")[-1])
            major, minor = torch.cuda.get_device_capability(device_id)
            device_arch = f"sm_{major}{minor}"
            supported_arches = set(torch.cuda.get_arch_list())
            if device_arch not in supported_arches:
                logger.warning(
                    "Requested %s (%s) is not supported by current PyTorch CUDA build (%s). Falling back to CPU.",
                    requested_device,
                    device_arch,
                    ", ".join(sorted(supported_arches)) if supported_arches else "unknown",
                )
                return "cpu"
        except Exception as exc:
            logger.warning(
                "Could not validate CUDA architecture for %s (%s). Falling back to CPU.",
                requested_device,
                exc,
            )
            return "cpu"

        return requested_device
        
    def load_model(self):
        raise NotImplementedError
        
    def unload_model(self):
        if self.model is not None:
            logger.info(f"Unloading model from {self.device}...")
            del self.model
            self.model = None
            
        if self.rembg_session is not None:
            del self.rembg_session
            self.rembg_session = None
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
    def process_image(self, image_path: str) -> Image.Image:
        """
        Removes background and resizes the foreground object.
        Uses a dedicated device for rembg to save VRAM on the main generation GPU.
        """
        logger.info(f"Processing image: {image_path} on {DEVICE_REMBG}")
        image = Image.open(image_path)
        
        if self.rembg_session is not None:
            # Remove background
            image = remove_background(image, self.rembg_session)
        
        # Resize foreground to fit the model's expected input
        image = resize_foreground(image, 0.85)
        
        return image
        
    def normalize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Normalizes the mesh for Looking Glass display:
        1. Centers the mesh at the origin (0, 0, 0).
        2. Scales the mesh to fit within a unit sphere (radius = 1).
        """
        logger.info("Normalizing mesh for Looking Glass...")
        
        # 1. Center at origin
        bounding_box = mesh.bounding_box.bounds
        center = (bounding_box[0] + bounding_box[1]) / 2.0
        mesh.apply_translation(-center)
        
        # 2. Scale to fit unit sphere
        max_distance = np.max(np.linalg.norm(mesh.vertices, axis=1))
        if max_distance > 0:
            scale_factor = 1.0 / max_distance
            scale_matrix = np.eye(4)
            scale_matrix[:3, :3] *= scale_factor
            mesh.apply_transform(scale_matrix)
            
        return mesh

    def generate(self, image_path: str, output_path: str, mc_resolution: int = 256):
        raise NotImplementedError


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


class TripoSRGenerator(BaseGenerator):
    """
    Handles the 3D generation pipeline using TripoSR.
    """
    def __init__(self, device: str = DEVICE_TRIPOSR):
        super().__init__(device)
        
    def load_model(self):
        if self.model is not None:
            return

        self.device = self._resolve_supported_device(self.device)
            
        logger.info(f"Loading TripoSR model to {self.device} in float16...")
        
        # VRAM Safety Check
        if self.device != "cpu":
            try:
                device_id = int(self.device.split(":")[-1])
                free_mem, total_mem = torch.cuda.mem_get_info(device_id)
                logger.info(f"Current free VRAM on {self.device}: {free_mem / 1024**3:.2f} GB")
                
                if free_mem < 4 * 1024**3: # Less than 4GB
                    logger.warning("Low VRAM detected. Flushing memory...")
                    gc.collect()
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Could not check VRAM: {e}")

        try:
            # Load model
            self.model = TSR.from_pretrained(
                TRIPOSR_MODEL_ID,
                config_name="config.yaml",
                weight_name="model.ckpt",
                local_files_only=LOAD_LOCAL_ONLY,
            )
            
            self.model.to(self.device)
            if self.device.startswith("cuda"):
                self.model.half()
            else:
                self.model.float()
            
            # Initialize rembg session on dedicated device
            os.environ["U2NET_HOME"] = os.path.expanduser("~/.u2net")
            self.rembg_session = rembg.new_session()

        except Exception as e:
            import traceback
            logger.error(f"❌ TripoSR Load failed: {traceback.format_exc()}")
            raise
        
    def generate(self, image_path: str, output_path: str, mc_resolution: int = 128):
        try:
            self.load_model()
            
            processed_image = self.process_image(image_path)
            
            image_array = np.array(processed_image).astype(np.float32) / 255.0
            image_array = image_array[:, :, :3] * image_array[:, :, 3:4] + (1 - image_array[:, :, 3:4]) * 0.5
            image_pil = Image.fromarray((image_array * 255.0).astype(np.uint8))
            
            logger.info("Running TripoSR model...")
            autocast_context = (
                torch.autocast(device_type='cuda', dtype=torch.float16)
                if self.device.startswith("cuda")
                else nullcontext()
            )
            with torch.no_grad():
                with autocast_context:
                    scene_codes = self.model([image_pil], device=self.device)
                
            logger.info(f"Extracting mesh (resolution={mc_resolution})...")
            meshes = self.model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mc_resolution)
            mesh = meshes[0]
            
            mesh = self.normalize_mesh(mesh)
            
            logger.info(f"Exporting mesh to {output_path}...")
            mesh.export(output_path, file_type='glb')
            
            logger.info("✅ TripoSR 3D generation complete!")
            return True
            
        except Exception as e:
            logger.error(f"❌ TripoSR Generation failed: {str(e)}")
            raise
        finally:
            logger.info("Running memory guard cleanup...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class StableFast3DGenerator(BaseGenerator):
    """
    Handles the 3D generation pipeline using StableFast3D.
    """
    def __init__(self, device: str = DEVICE_STABLEFAST3D):
        super().__init__(device)
        
    def load_model(self):
        if self.model is not None:
            return

        self.device = self._resolve_supported_device(self.device)
            
        logger.info(f"Loading StableFast3D model to {self.device}...")
        
        # VRAM Safety Check
        if self.device != "cpu":
            try:
                device_id = int(self.device.split(":")[-1])
                if device_id == 1:
                    logger.info("Flushing GPU memory before StableFast3D load on cuda:1...")
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.synchronize(device_id)

                free_mem, total_mem = torch.cuda.mem_get_info(device_id)
                free_gb = free_mem / 1024**3
                total_gb = total_mem / 1024**3
                logger.info(f"Current free VRAM on {self.device}: {free_gb:.2f} GB / {total_gb:.2f} GB")
                
                if device_id == 1 and free_gb < 5.0:
                    raise RuntimeError(
                        f"Insufficient VRAM on cuda:1 ({free_gb:.2f} GB free). "
                        "StableFast3D load aborted; at least 5.0 GB free is required."
                    )
            except Exception as e:
                logger.error(f"VRAM guard failed: {e}")
                raise

        try:
            model_path = str(_ensure_sf3d_weights(SF3D_MODEL_PATH))
            logger.info(f"Using forced local model path: {model_path}")
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"StableFast3D local snapshot path not found: {model_path}")

            # Load model using SF3D class from the cloned repo
            self.model = SF3D.from_pretrained(
                model_path,
                config_name="config.yaml",
                weight_name="model.safetensors",
                local_files_only=True,
                trust_remote_code=True,
            )
            
            self.model.to(self.device)
            self.model.eval()

            if hasattr(self.model, "image_estimator"):
                self.model.image_estimator = None
                logger.info("Disabled SF3D image_estimator to reduce VRAM usage.")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Initialize rembg session
            self.rembg_session = rembg.new_session()
            
        except Exception as e:
            import traceback
            logger.error(f"❌ Failed to load StableFast3D: {traceback.format_exc()}")
            logger.error("Note: StableFast3D requires access to the gated repository on Hugging Face.")
            logger.error("If running locally, ensure weights are downloaded to cache.")
            raise

    def _generate_geometry_only_mesh(self, processed_image: Image.Image) -> trimesh.Trimesh:
        """
        Fallback path for low-VRAM / unstable texture baking kernels.
        Generates geometry only and skips UV/texture baking.
        """
        mask_cond, rgb_cond = self.model.prepare_image(processed_image)
        rgb_cond = rgb_cond.unsqueeze(0).unsqueeze(0)
        mask_cond = mask_cond.unsqueeze(0).unsqueeze(0)
        batch_size = 1

        c2w_cond = sf3d_default_cond_c2w(self.model.cfg.default_distance).to(self.model.device)
        intrinsic, intrinsic_normed_cond = sf3d_create_intrinsic_from_fov_deg(
            self.model.cfg.default_fovy_deg,
            self.model.cfg.cond_image_size,
            self.model.cfg.cond_image_size,
        )

        batch = {
            "rgb_cond": rgb_cond,
            "mask_cond": mask_cond,
            "c2w_cond": c2w_cond.view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1),
            "intrinsic_cond": intrinsic.to(self.model.device).view(1, 1, 3, 3).repeat(batch_size, 1, 1, 1),
            "intrinsic_normed_cond": intrinsic_normed_cond.to(self.model.device).view(1, 1, 3, 3).repeat(batch_size, 1, 1, 1),
        }

        with torch.no_grad():
            scene_codes, _ = self.model.get_scene_codes(batch)
            meshes = self.model.triplane_to_meshes(scene_codes)

        if not meshes:
            raise ValueError("SF3D geometry fallback returned no mesh")

        mesh = meshes[0]
        vertices = mesh.v_pos.detach().float().cpu().numpy()
        faces = mesh.t_pos_idx.detach().cpu().numpy()

        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            
    def generate(self, image_path: str, output_path: str, mc_resolution: int = 256):
        try:
            self.load_model()
            
            # Process image using tsr.utils (compatible with sf3d)
            # Remove background and resize
            input_image = Image.open(image_path).convert("RGBA")
            processed_image = remove_background(input_image, self.rembg_session)
            processed_image = resize_foreground(processed_image, 0.85)
            
            logger.info("Running StableFast3D model...")
            mesh = self._generate_geometry_only_mesh(processed_image)
                
            if mesh is not None:
                
                # Normalize mesh for display
                # Note: normalize_mesh expects a trimesh object
                if isinstance(mesh, trimesh.Trimesh):
                    mesh = self.normalize_mesh(mesh)
                    
                    logger.info(f"Exporting mesh to {output_path}...")
                    mesh.export(output_path, file_type='glb')
                else:
                    logger.error("Output is not a valid Trimesh object.")
                    raise ValueError("Invalid output format")
            else:
                logger.error("No mesh generated.")
                raise ValueError("Model failed to generate mesh")
            
            logger.info("✅ StableFast3D generation complete!")
            return True
            
        except Exception as e:
            logger.error(f"❌ StableFast3D Generation failed: {str(e)}")
            raise
        finally:
            logger.info("Running memory guard cleanup...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
