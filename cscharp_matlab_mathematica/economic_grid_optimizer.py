import torch

# 1. Initialize GPU configuration
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available!")
device = torch.device("cuda:0")

# Set up a 1,000 x 1,000 grid to hit exactly 1,000,000 distinct simulations
grid_size = 1000
num_simulations = grid_size * grid_size

print(f"Generating 1,000,000 economic parameter states on NVIDIA 5060...")

# 2. Define fixed baseline parameters
beta_val = 0.95  # Discount factor
pz_val = 10.0   # Production scale at z_bar

# 3. Create parameter arrays for the grid search
# theta ranges linearly from 0.01 to 0.99 (1000 steps)
# tau ranges linearly from 0.05 to 0.50 (1000 steps)
theta_steps = torch.linspace(0.01, 0.99, grid_size, device=device)
tau_steps = torch.linspace(0.05, 0.50, grid_size, device=device)

# Build the 2D grid matrix combinations
theta_grid, tau_grid = torch.meshgrid(theta_steps, tau_steps, indexing="ij")

# Flatten the grids to form 1,000,000 parallel parameter tracks
thetas = theta_grid.reshape(-1)
taus = tau_grid.reshape(-1)

# 4. Construct the structural Matrix A and vector b blocks for all 1,000,000 systems
# Matrix A coefficients for:
# Row 1: (1 - beta*(1-theta))*V0 - beta*theta*Vz = -beta*theta*tau*pz
# Row 2: -beta*V0 + Vz = pz
a11 = 1.0 - beta_val * (1.0 - thetas)
a12 = -beta_val * thetas
a21 = torch.full((num_simulations,), -beta_val, device=device)
a22 = torch.ones(num_simulations, device=device)

# Stack elements into a [1000000, 2, 2] matrix batch
A_batch = torch.stack([
    torch.stack([a11, a12], dim=1),
    torch.stack([a21, a22], dim=1)
], dim=1)

# Vector b components:
b1 = -beta_val * thetas * taus * pz_val
b2 = torch.full((num_simulations,), pz_val, device=device)

# Stack elements into a [1000000, 2, 1] vector batch
b_batch = torch.stack([b1, b2], dim=1).unsqueeze(-1)

# 5. Execute parallel GPU solving
print(f"Launching batch optimization solver across the parameter grid...")
torch.cuda.synchronize()
start_time = time() if 'time' in locals() else None

# Solve all 1,000,000 linear matrix arrays simultaneously
solutions = torch.linalg.solve(A_batch, b_batch)

torch.cuda.synchronize()

# Extract value vectors
V0_results = solutions[:, 0, 0]
Vz_results = solutions[:, 1, 0]

# 6. Apply economic strategy boundary constraints
# Condition: Discard regimes where frictional costs degrade value functions below a minimum threshold
min_acceptable_value = 25.0
valid_mask = (V0_results >= min_acceptable_value) & (Vz_results > V0_results)

# Count valid configurations
num_valid = torch.sum(valid_mask).item()

# 7. Locate the maximum value optimization coordinate
max_val, max_idx = torch.max(V0_results * valid_mask.float(), dim=0)

print("\n--- Value Function Optimization Summary ---")
print(f"Total Simulation States Computed : {num_simulations:,}")
print(f"States Meeting Value Constraints  : {num_valid:,} ({num_valid/num_simulations*100:.1f}%)")
print(f"Optimal V0 Value Discovered      : {max_val.item():.4f}")
print(f"Optimal Coordinate Parameters    : Theta = {thetas[max_idx].item():.4f}, Tau = {taus[max_idx].item():.4f}")


# --- NEW EXPORT SECTION ---
print("\nExporting optimization landscape to binary format...")

# Move tensors back to host CPU memory just for the file-writing process
export_data = {
    "thetas": thetas.cpu(),
    "taus": taus.cpu(),
    "V0_results": V0_results.cpu(),
    "Vz_results": Vz_results.cpu(),
    "valid_mask": valid_mask.cpu()
}

# Save as a highly compressed, rapid-load PyTorch binary file
output_path = r"C:\csharp_code_analysis\grid_optimization_landscape.pt"
torch.save(export_data, output_path)

print(f"Landscape successfully saved to: {output_path}")
print("You can reload this dataset instantly in any script using: torch.load(path)")


# =====================================================================
# 3D SURFACE PLOTTER SECTION (APPEND TO THE BOTTOM OF YOUR CODE)
# =====================================================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("\nPreparing 3D surface plot landscape...")

# 1. Move data back to CPU and reshape it back into the original 2D Grid shapes
# We also apply the valid mask to set invalid regimes to a flat minimum value (e.g., 0)
masked_V0 = (V0_results * valid_mask.float()).reshape(grid_size, grid_size).cpu().numpy()
theta_plot = theta_grid.cpu().numpy()
tau_plot = tau_grid.cpu().numpy()

# 2. Downsample the grid (Take every 10th point) for fluid 3D rendering performance
stride = 10
X = theta_plot[::stride, ::stride]
Y = tau_plot[::stride, ::stride]
Z = masked_V0[::stride, ::stride]

# 3. Initialize the Matplotlib Plot Objects
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 4. Draw the surface landscape
# 'cmap="viridis"' applies an intuitive dark-blue to bright-yellow color gradient
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# 5. Plot a bright red star marker exactly at your calculated optimal coordinate
opt_theta = thetas[max_idx].item()
opt_tau = taus[max_idx].item()
opt_V0 = max_val.item()

ax.scatter([opt_theta], [opt_tau], [opt_V0], color='red', s=200, marker='*', 
           label=f'Optimal (V0={opt_V0:.2f})', depthshade=False)

# 6. Customize chart labels and visual perspective angles
ax.set_title("Value Function (V0) Strategy Optimization Landscape", fontsize=14, pad=20)
ax.set_xlabel("Transition Probability (\u03b8)", fontsize=11, labelpad=10)
ax.set_ylabel("Friction / Tax Scale (\u03c4)", fontsize=11, labelpad=10)
ax.set_zlabel("Value Function (V0)", fontsize=11, labelpad=10)

# Add a color bar legend scale to the side
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
ax.legend(loc="upper left")

# Adjust camera view angle (elevation, azimuth) for an optimal perspective look
ax.view_init(elev=30, azim=-135)

print("Displaying interactive 3D surface model plot...")
plt.tight_layout()
plt.show()



