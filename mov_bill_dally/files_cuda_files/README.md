# NVIDIA DPX Smith-Waterman CUDA Examples
### Based on Bill Dally / NVIDIA Bioinformatics DPX Work

---

## Files

| File | Description |
|------|-------------|
| `smith_waterman_cuda.cu` | Simple anti-diagonal wavefront Smith-Waterman with DPX path |
| `sw_dpx_batch.cu` | Advanced packed 16-bit DPX batch aligner (one block per query) |
| `Makefile` | Build system for both files |

---

## Quick Start

```bash
# 1. Check your GPU
make info

# 2. Build both
make

# 3. Run
./sw_simple
./sw_dpx_batch
```

---

## Architecture Targets

| GPU | Architecture | sm_ | DPX? |
|-----|-------------|-----|------|
| RTX 5060 | Blackwell | sm_120 | ✓ |
| H100 | Hopper | sm_90 | ✓ |
| RTX 4090 | Ada Lovelace | sm_89 | ✗ (fallback) |
| A100 | Ampere | sm_80 | ✗ (fallback) |

The default `Makefile` targets `sm_90,sm_100` — produces a portable binary
that uses DPX on Hopper/Blackwell and falls back on older GPUs.

For the RTX 5060 natively:
```bash
make ARCH="-arch=sm_100a"
```

---

## Requirements

- CUDA Toolkit 12.8+
- NVIDIA driver 570+
- C++17

---

## References

- NVIDIA DPX Blog: https://developer.nvidia.com/blog/boosting-dynamic-programming-performance-using-nvidia-hopper-gpu-dpx-instructions/
- Clara Parabricks (production genomics): https://developer.nvidia.com/clara-parabricks
- CUDASW++ 4.0 (open-source DPX SW): https://github.com/solodon4/CUDASW
