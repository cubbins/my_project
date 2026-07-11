#!/bin/bash

# Full build + analysis pipeline for fastWalshTransform sample

SRC_CU="fastWalshTransform.cu"
SRC_CPU="fastWalshTransform_gold.cpp"
INCLUDE_DIR="$HOME/cuda-samples/Common"
ARCH="sm_120"

echo "=== Building executable ==="
nvcc -arch=$ARCH \
    $SRC_CU \
    $SRC_CPU \
    -I $INCLUDE_DIR \
    -o fastWalshTransform

echo "Executable built: ./fastWalshTransform"

echo "=== Generating PTX ==="
nvcc -arch=$ARCH -ptx \
    $SRC_CU \
    -I $INCLUDE_DIR \
    -o fastWalshTransform.ptx

echo "PTX generated: fastWalshTransform.ptx"

echo "=== Generating CUBIN ==="
nvcc -arch=$ARCH -cubin \
    $SRC_CU \
    -I $INCLUDE_DIR \
    -o fastWalshTransform.cubin

echo "CUBIN generated: fastWalshTransform.cubin"

echo "=== Generating SASS (nvdisasm) ==="

# nvdisasm fastWalshTransform.cubin > fastWalshTransform.sass
# nvdisasm -hex fastWalshTransform.cubin > fastWalshTransform_with_hex.sass

nvdisasm \
  -hex \
  --print-instruction-encoding \
  --separate-functions \
  fastWalshTransform.cubin \
  > fastWalshTransform_hex.sass




echo "SASS generated: fastWalshTransform.sass"

echo "=== Done ==="
echo "Run executable with: ./fastWalshTransform"
