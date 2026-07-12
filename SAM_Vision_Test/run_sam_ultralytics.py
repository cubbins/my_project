import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import SAM

# 1. Initialize Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Target Hardware Device: {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

# 2. Load the pre-compiled SAM2 model (automatically downloads original weights)
print("[*] Loading SAM2 via Ultralytics Framework into RTX 5060 VRAM...")
model = SAM("sam2_t.pt")  # 't' stands for tiny, matching your original script

# 3. Create a Dummy Image for Verification
print("[*] Generating synthetic test image...")
image_np = np.zeros((600, 600, 3), dtype=np.uint8) + 128  # Gray background
cv2.rectangle(image_np, (200, 200), (400, 400), (255, 0, 0), -1)  # Blue target square
image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

# 4. Task A: Automatic Mask Generation (Detect Everything)
print("[*] Task A: Running Automatic Mask Generation...")
auto_results = model(image_rgb, device=device)
print(f"    - Found {len(auto_results[0].masks)} distinct objects automatically.")

# 5. Task B: Point-Prompted Segmentation (Isolate the specific square)
print("[*] Task B: Running Point-Prompted Segmentation...")
# Click coordinate [300, 300] exactly inside the blue square
prompt_results = model(image_rgb, points=[300, 300], labels=[1], device=device)

# 6. Visualize and Save the Results
print("[*] Rendering visual plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Original Image
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Plot 2: Auto Mask Generation Output
axes[1].imshow(image_rgb)
if auto_results[0].masks is not None:
    auto_overlay = np.zeros_like(image_rgb)
    for mask in auto_results[0].masks.data.cpu().numpy():
        color = np.random.randint(0, 255, size=3)
        auto_overlay[mask.astype(bool)] = color
    axes[1].imshow(auto_overlay, alpha=0.5)
axes[1].set_title(f"Auto-Detected Blocks ({len(auto_results[0].masks)})")
axes[1].axis('off')

# Plot 3: Point-Prompted Output
axes[2].imshow(image_rgb)
if prompt_results[0].masks is not None:
    prompt_mask = prompt_results[0].masks.data.cpu().numpy()[0]
    prompted_overlay = np.zeros_like(image_rgb)
    prompted_overlay[prompt_mask.astype(bool)] = [0, 255, 0]  # Color it Green
    axes[2].imshow(prompted_overlay, alpha=0.5)
axes[2].scatter([300], [300], color='red', marker='*', s=200, label='Prompt Click')
axes[2].set_title("Targeted Point Mask")
axes[2].legend()
axes[2].axis('off')

# Save to file
output_filename = "sam2_ultralytics_result.png"
plt.tight_layout()
plt.savefig(output_filename)
print(f"[🏆] SUCCESS! Pipeline complete. Result saved to: '{os.path.abspath(output_filename)}'")
plt.show()
