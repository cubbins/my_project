import torch
import matlab.engine

# 1. Generate an execution variable tensor directly inside your PyTorch environment
# (e.g., simulating a batch of parameter combinations generated on your GPU)
cuda_tensor = torch.tensor([[2.0, 1.0], [1.0, -1.0]], device="cuda")

# 2. Safely step data down from VRAM to Host CPU memory space as a standard array
cpu_numpy = cuda_tensor.cpu().numpy()

# 3. Cast the matrix straight into the MATLAB double interface layer
matlab_matrix = matlab.double(cpu_numpy.tolist())

print(f"Data ready for MATLAB execution context: {matlab_matrix}")
