import torch
import matlab.engine

# 1. Start the MATLAB session
print("Connecting to MATLAB Engine...")
eng = matlab.engine.start_matlab()

# 2. Simulate parameters generated via PyTorch (e.g., Matrix A and vector b)
A_tensor = torch.tensor([[row1_V0, row1_Vz], [row2_V0, row2_Vz]] if 'row1_V0' in locals() else [[0.24, -0.19], [-0.95, 1.0]], device="cuda")
b_tensor = torch.tensor([[-0.285], [10.0]], device="cuda")

# 3. Convert PyTorch tensors to CPU-bound lists for the MATLAB engine
A_matlab = matlab.double(A_tensor.cpu().numpy().tolist())
b_matlab = matlab.double(b_tensor.cpu().numpy().tolist())

print("Executing structural matrix division inside MATLAB...")
# Execute the calculation via MATLAB's backslash engine matrix solver
raw_solution = eng.mldivide(A_matlab, b_matlab)

# 4. Read data back into PyTorch and push it directly onto your GPU
# Convert the MATLAB array object back into a standard Python nested list structure
python_list = [list(row) for row in raw_solution]

# Load it straight into PyTorch as a float32 CUDA Tensor
# result_tensor = torch.tensor(python_list, dtype=torch.float32, device="cuda")

# This runs entirely inside your NVIDIA 5060 GPU cores instantly
result_tensor = torch.linalg.solve(A_tensor, b_tensor)

print("\n--- Optimization Pipeline Complete ---")
print(f"Resulting Tensor Type : {result_tensor.type()}")
print(f"Resulting Tensor Device : {result_tensor.device}")
print(f"Calculated Solutions Vector:\n{result_tensor}")

eng.quit()
