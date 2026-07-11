#!/bin/bash

# Full build + analysis pipeline for fastWalshTransform sample

SRC_CU="fastWalshTransform.cu"
SRC_CPU="fastWalshTransform_gold.cpp"
INCLUDE_DIR="$HOME/cuda-samples/Common"
ARCH="sm_120"

echo "=== Building debug executable (for cuda-gdb) ==="
nvcc -arch=$ARCH \
    -G \
    -g \
    $SRC_CU \
    $SRC_CPU \
    -I $INCLUDE_DIR \
    -o fastWalshTransform

echo "Executable built: ./fastWalshTransform"

echo "=== Generating PTX (maximum detail, optimized) ==="
nvcc -arch=$ARCH -ptx \
    -I $INCLUDE_DIR \
    -lineinfo \
    -src-in-ptx \
    -Xptxas -v \
    $SRC_CU \
    -o fastWalshTransform.ptx

echo "PTX generated: fastWalshTransform.ptx"

echo "=== Generating CUBIN (optimized, for SASS disassembly) ==="
nvcc -arch=$ARCH -cubin \
    $SRC_CU \
    -I $INCLUDE_DIR \
    -o fastWalshTransform.cubin

echo "CUBIN generated: fastWalshTransform.cubin"

echo "=== Generating debug CUBIN (matches debuggable executable) ==="
nvcc -arch=$ARCH -cubin \
    -G \
    -g \
    $SRC_CU \
    -I $INCLUDE_DIR \
    -o fastWalshTransform_debug.cubin

echo "Debug CUBIN generated: fastWalshTransform_debug.cubin"

echo "=== Generating SASS from optimized CUBIN (nvdisasm) ==="
nvdisasm \
  -hex \
  --print-instruction-encoding \
  --separate-functions \
  fastWalshTransform.cubin \
  > fastWalshTransform_hex.sass

echo "SASS generated: fastWalshTransform_hex.sass"

echo "=== Done ==="
echo "Debug session:    cuda-gdb -x trace_kernel.py ./fastWalshTransform"
echo "Optimized run:    ./fastWalshTransform"
