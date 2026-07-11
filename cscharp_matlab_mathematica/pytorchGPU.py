import torch

# 1. Keep your parameters completely native to CUDA from the start
# (Let's define the matrix elements from your economic equation)
A = torch.tensor([[0.24, -0.19], [-0.95, 1.0]], dtype=torch.float32, device="cuda")
b = torch.tensor([[-0.285], [10.0]], dtype=torch.float32, device="cuda")

# 2. Compute the exact matrix solution vector directly inside your 5060 GPU cores
result_tensor = torch.linalg.solve(A, b)

print("--- 100% Native GPU Math Complete ---")
print(result_tensor)
