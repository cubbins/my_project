import torch
import time

# 1. Ensure CUDA is fully initialized
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Check your torchgpu environment setup!")

device = torch.device("cuda:0")
num_simulations = 1_000_000

print(f"Initializing {num_simulations:,} simultaneous equations in VRAM...")

# 2. Set up precise CUDA hardware timing events
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 3. Generate randomized economic parameter matrices directly on the GPU
# We are creating a batch of 1,000,000 distinct 2x2 systems (A) and 2x1 vectors (b)
# This simulates 1,000,000 different combinations of beta, theta, tau, and pz
torch.manual_seed(42)  # For reproducible randomness

# Matrix A shape: [1000000, 2, 2]
A_batch = torch.randn(num_simulations, 2, 2, device=device, dtype=torch.float32)
# Ensure matrices are invertible by adding an identity scaling factor
A_batch += torch.eye(2, device=device).unsqueeze(0) * 2.0 

# Vector b shape: [1000000, 2, 1]
b_batch = torch.randn(num_simulations, 2, 1, device=device, dtype=torch.float32)

print("Memory allocation complete. Warming up GPU cores...")
# Warm-up pass to let the CUDA driver compile underlying kernels
_ = torch.linalg.solve(A_batch[:100], b_batch[:100])
torch.cuda.synchronize()

print(f"Launching batch execution of {num_simulations:,} equations on your NVIDIA 5060...")

# 4. Start timing and execute
start_event.record()

# This single line executes all 1,000,000 matrix solutions in parallel
solutions_batch = torch.linalg.solve(A_batch, b_batch)

end_event.record()

# 5. Force CPU to wait until the GPU completely finishes the workload
torch.cuda.synchronize()

# Calculate total elapsed time
elapsed_time_ms = start_event.elapsed_time(end_event)
equations_per_second = num_simulations / (elapsed_time_ms / 1000.0)

print("\n--- GPU Stress Test Complete ---")
print(f"Total Execution Time      : {elapsed_time_ms:.2f} ms")
print(f"Processing Throughput     : {equations_per_second:,.0f} equations/sec")
print(f"Output Tensor Shape       : {list(solutions_batch.shape)}")
print(f"Output Storage Device     : {solutions_batch.device}")

# Display a quick snapshot of the first 2 computed result vectors
print("\nSnapshot of first two solutions:")
print(f"Simulation #1 Vector:\n{solutions_batch[0]}")
print(f"Simulation #2 Vector:\n{solutions_batch[1]}")
