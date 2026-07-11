#!/bin/bash

# Build script for fastWalshTransform sample
# Compiles both the CUDA file and the CPU reference implementation

nvcc -arch=sm_120 \
    fastWalshTransform.cu \
    fastWalshTransform_gold.cpp \
    -I ~/cuda-samples/Common \
    -o fastWalshTransform

echo "Build complete. Run with: ./fastWalshTransform"
