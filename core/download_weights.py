import os
import logging
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DownloadWeights")

MODELS = [
    "stabilityai/TripoSR",
    "stabilityai/stable-fast-3d"
]

def download_models():
    for model_id in MODELS:
        logger.info(f"Downloading {model_id}...")
        try:
            # snapshot_download saves to the default cache directory
            # (~/.cache/huggingface/hub/...)
            path = snapshot_download(repo_id=model_id, token=True)
            logger.info(f"Successfully downloaded {model_id} to {path}")
            
            # Verify files exist and print sizes of .safetensors or .ckpt files
            if os.path.exists(path) and len(os.listdir(path)) > 0:
                logger.info(f"✅ Verified files exist in {path}")
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith(".safetensors") or file.endswith(".ckpt"):
                            file_path = os.path.join(root, file)
                            size_mb = os.path.getsize(file_path) / (1024 * 1024)
                            logger.info(f"  - {file}: {size_mb:.2f} MB")
            else:
                logger.error(f"❌ Directory {path} is empty or does not exist.")
        except Exception as e:
            logger.error(f"❌ Failed to download {model_id}: {e}")

if __name__ == "__main__":
    download_models()
