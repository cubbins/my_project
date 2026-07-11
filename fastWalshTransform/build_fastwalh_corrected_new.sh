#!/bin/bash

# ============================================================
# Full build + analysis pipeline for fastWalshTransform
# Enhanced for:
#   - CUDA-GDB debugging
#   - low optimization (accurate tracing)
#   - PTX/SASS analysis
# ============================================================

SRC_CU="fastWalshTransform.cu"
SRC_CPU="fastWalshTransform_gold.cpp"
INCLUDE_DIR="$HOME/cuda-samples/Common"
ARCH="sm_120"

# ============================================================
# 1. DEBUG BUILD (LOW OPTIMIZATION, BEST FOR cuda-gdb)
# ============================================================
echo "=== Building DEBUG executable (low optimization, cuda-gdb ready) ==="

nvcc -arch=$ARCH \
    -G \
    -g \
    -O0 \
    -Xptxas -O0 \
    -Xcompiler -O0 \
    -lineinfo \
    $SRC_CU \
    $SRC_CPU \
    -I $INCLUDE_DIR \
    -o fastWalshTransform_debug

echo "Executable built: ./fastWalshTransform_debug"

# ============================================================
# 2. OPTIMIZED BUILD (REFERENCE / PERFORMANCE)
# ============================================================
echo "=== Building OPTIMIZED executable ==="

nvcc -arch=$ARCH \
    -O3 \
    $SRC_CU \
    $SRC_CPU \
    -I $INCLUDE_DIR \
    -o fastWalshTransform

echo "Executable built: ./fastWalshTransform"

# ============================================================
# 3. DEBUG PTX (MATCHES DEBUG EXECUTION)
# ============================================================
echo "=== Generating DEBUG PTX (low optimization, full mapping) ==="

nvcc -arch=$ARCH -ptx \
    -G \
    -O0 \
    -Xptxas -O0 \
    -lineinfo \
    -src-in-ptx \
    $SRC_CU \
    -I $INCLUDE_DIR \
    -o fastWalshTransform_debug.ptx

echo "PTX generated: fastWalshTransform_debug.ptx"

# ============================================================
# 4. OPTIMIZED PTX (for comparison)
# ============================================================
echo "=== Generating OPTIMIZED PTX ==="

nvcc -arch=$ARCH -ptx \
    -O3 \
    -lineinfo \
    -src-in-ptx \
    -Xptxas -v \
    $SRC_CU \
    -I $INCLUDE_DIR \
    -o fastWalshTransform.ptx

echo "PTX generated: fastWalshTransform.ptx"

# ============================================================
# 5. DEBUG CUBIN (MATCHES cuda-gdb)
# ============================================================
echo "=== Generating DEBUG CUBIN ==="

nvcc -arch=$ARCH -cubin \
    -G \
    -O0 \
    -Xptxas -O0 \
    $SRC_CU \
    -I $INCLUDE_DIR \
    -o fastWalshTransform_debug.cubin

echo "Debug CUBIN generated: fastWalshTransform_debug.cubin"

# ============================================================
# 6. OPTIMIZED CUBIN (FOR SASS ANALYSIS)
# ============================================================
echo "=== Generating OPTIMIZED CUBIN ==="

nvcc -arch=$ARCH -cubin \
    -O3 \
    $SRC_CU \
    -I $INCLUDE_DIR \
    -o fastWalshTransform.cubin

echo "CUBIN generated: fastWalshTransform.cubin"

# ============================================================
# 7. SASS DISASSEMBLY (OPTIMIZED)
# ============================================================
echo "=== Generating SASS (optimized) ==="

nvdisasm \
  -hex \
  --print-instruction-encoding \
  --separate-functions \
  fastWalshTransform.cubin \
  > fastWalshTransform_hex.sass

echo "SASS generated: fastWalshTransform_hex.sass"

# ============================================================
# 8. SASS DISASSEMBLY (DEBUG — VERY USEFUL!)
# ============================================================
echo "=== Generating SASS (debug version) ==="

nvdisasm \
  --print-line-info \
  --separate-functions \
  fastWalshTransform_debug.cubin \
  > fastWalshTransform_debug.sass

echo "Debug SASS generated: fastWalshTransform_debug.sass"

# ============================================================
# DONE
# ============================================================
echo "=== Done ==="
echo ""
echo "DEBUG session:"
echo "  cuda-gdb -x trace_kernel_json_safe_deferred.py ./fastWalshTransform_debug"
echo ""
echo "Optimized run:"
echo "  ./fastWalshTransform"