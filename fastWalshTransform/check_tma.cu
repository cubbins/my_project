#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int dev = 0;
    int val = 0;
    cudaDeviceGetAttribute(&val, cudaDevAttrTmaSupported, dev);
    printf("TMA Supported: %d\n", val);
    return 0;
}
