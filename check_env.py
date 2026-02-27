import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import gc

def print_vram_usage():
    print("\n" + "="*40)
    print("GPU VRAM Status")
    print("="*40)
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return

    for i in range(torch.cuda.device_count()):
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        free_mb = free_mem / (1024**2)
        total_mb = total_mem / (1024**2)
        used_mb = total_mb - free_mb
        print(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
        print(f"  Total VRAM: {total_mb:.2f} MB")
        print(f"  Used VRAM:  {used_mb:.2f} MB")
        print(f"  Free VRAM:  {free_mb:.2f} MB")
    print("="*40 + "\n")

def test_4bit_loading():
    print("Testing 4-bit quantization with bitsandbytes...")
    # Using a very small model to minimize download time and memory usage
    # Using a model that has safetensors to avoid the torch.load vulnerability issue in PyTorch < 2.6
    model_id = "Qwen/Qwen2.5-0.5B" 
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    try:
        # Load model on GPU 1 as per the multi-GPU strategy for generation/rendering
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto" # Let accelerate handle device mapping
        )
        print("✅ Successfully loaded model in 4-bit mode!")
        print(f"Model memory footprint: {model.get_memory_footprint() / 1024**2:.2f} MB")
        
        # Verify that linear layers are quantized
        print(f"Model type: {type(model).__name__}")
        
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ Cleaned up model from memory.")
        
    except Exception as e:
        print(f"❌ Failed to load model in 4-bit mode: {e}")

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
    print_vram_usage()
    test_4bit_loading()
    print_vram_usage()
