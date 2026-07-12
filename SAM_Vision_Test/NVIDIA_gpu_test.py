import torch
import sys
import time

def test_rtx5060_environment():
    print("=" * 50)
    print("         NVIDIA RTX 5060 WINDOWS CONDA TEST         ")
    print("=" * 50)
    
    # 1. Basic System Diagnostics
    print(f"[*] Python Version: {sys.version.split()[0]}")
    print(f"[*] PyTorch Version: {torch.__version__}")
    
    # 2. CUDA Availability
    cuda_available = torch.cuda.is_available()
    print(f"[*] CUDA Available: {'✅ YES' if cuda_available else '❌ NO'}")
    
    if not cuda_available:
        print("\n[!] ERROR: PyTorch cannot see your GPU.")
        print("Please verify your Windows NVIDIA driver is updated to 561+.")
        return

    # 3. Hardware / Blackwell Architecture Detection
    gpu_name = torch.cuda.get_device_name(0)
    current_device = torch.cuda.current_device()
    arch_list = torch.cuda.get_arch_list()
    
    print(f"[*] GPU Device Name: {gpu_name}")
    print(f"[*] Active Device ID: {current_device}")
    print(f"[*] Compiled Architectures: {arch_list}")
    
    # Check for Blackwell 'sm_120' support
    if 'sm_120' in arch_list:
        print("[*] Blackwell Architecture Support: ✅ Verified (sm_120 present)")
    else:
        print("[!] Warning: 'sm_120' not found in architecture list. If computation fails, switch to PyTorch Nightly.")

    # 4. Actual Compute Stress Test (Matrix Multiplication)
    print("\n[*] Initializing Matrix Multiplication Compute Test...")
    try:
        # Create large random matrices directly on the RTX 5060 VRAM
        device = torch.device("cuda")
        size = 5000  # 5000x5000 matrices
        
        print(f"    - Allocating two {size}x{size} tensors on GPU VRAM...")
        x = torch.randn(size, size, device=device, dtype=torch.float32)
        y = torch.randn(size, size, device=device, dtype=torch.float32)
        
        print("    - Running matrix multiplication (X * Y)...")
        start_time = time.time()
        
        # Warmup and execute
        result = torch.matmul(x, y)
        torch.cuda.synchronize()  # Force execution sync to accurately measure time
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"    - Compute completed successfully in {elapsed:.4f} seconds!")
        print(f"    - VRAM Sample Matrix Shape: {result.shape}")
        print("\n🏆 STATUS: NVIDIA RTX 5060 is fully operational in this Conda environment!")
        
    except Exception as e:
        print(f"\n❌ COMPUTE ERROR: Matrix multiplication failed.")
        print(f"Details: {str(e)}")
        print("\n[Troubleshooting Tip]: If you see an illegal memory or kernel error, the current Windows PyTorch build lacks native binaries for the 5060. Fix this by running: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128")

if __name__ == "__main__":
    test_rtx5060_environment()
