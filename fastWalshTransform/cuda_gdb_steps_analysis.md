# CUDA-GDB Step Analysis Report

- Kernel: `fwtBatch2Kernel`
- Source file: `/home/cubbi/cuda-samples/Samples/5_Domain_Specific/fastWalshTransform/fastWalshTransform_kernel.cuh`
- Entry source line: `118`
- Steps parsed: `40`

## Registers that changed most often

| Register | Count |
|---|---:|
| pc | 39 |
| R6 | 5 |
| R7 | 2 |
| R0 | 1 |

## Step 1

- PC: `0x9010f2cf0`
- Source line: `118`
- Source text: `const int pos = blockIdx.x * blockDim.x + threadIdx.x;`
- Current instruction: `MOV R0,R0`

No deltas recorded for this step.

## Step 2

- PC: `0x9010f2d00`
- Source line: `118`
- Source text: `const int pos = blockIdx.x * blockDim.x + threadIdx.x;`
- Current instruction: `LDC R6,c[0x0][0x360]`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2cf0` | `0x9010f2d00` |

## Step 3

- PC: `0x9010f2d10`
- Source line: `118`
- Source text: `const int pos = blockIdx.x * blockDim.x + threadIdx.x;`
- Current instruction: `MOV R6,R6`

| Register | Old | New |
|---|---|---|
| R6 | `0x0` | `0x100` |
| pc | `0x9010f2d00` | `0x9010f2d10` |

## Step 4

- PC: `0x9010f2d20`
- Source line: `118`
- Source text: `const int pos = blockIdx.x * blockDim.x + threadIdx.x;`
- Current instruction: `IMAD R0,R0,R6,RZ`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2d10` | `0x9010f2d20` |

## Step 5

- PC: `0x9010f2d30`
- Source line: `118`
- Source text: `const int pos = blockIdx.x * blockDim.x + threadIdx.x;`
- Current instruction: `S2R R6,SR_TID.X`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2d20` | `0x9010f2d30` |

## Step 6

- PC: `0x9010f2d40`
- Source line: `118`
- Source text: `const int pos = blockIdx.x * blockDim.x + threadIdx.x;`
- Current instruction: `MOV R6,R6`

| Register | Old | New |
|---|---|---|
| R6 | `0x100` | `0x1` |
| pc | `0x9010f2d30` | `0x9010f2d40` |

## Step 7

- PC: `0x9010f2d50`
- Source line: `118`
- Source text: `const int pos = blockIdx.x * blockDim.x + threadIdx.x;`
- Current instruction: `IADD R0,R0,R6`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2d40` | `0x9010f2d50` |

## Step 8

- PC: `0x9010f2d60`
- Source line: `118`
- Source text: `const int pos = blockIdx.x * blockDim.x + threadIdx.x;`
- Current instruction: `MOV R0,R0`

| Register | Old | New |
|---|---|---|
| R0 | `0x0` | `0x1` |
| pc | `0x9010f2d50` | `0x9010f2d60` |

## Step 9

- PC: `0x9010f2d70`
- Source line: `None`
- Source text: `None`
- Current instruction: `LDC R6,c[0x0][0x360]`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2d60` | `0x9010f2d70` |

## Step 10

- PC: `0x9010f2d80`
- Source line: `119`
- Source text: `const int N   = blockDim.x * gridDim.x * 4;`
- Current instruction: `MOV R6,R6`

| Register | Old | New |
|---|---|---|
| R6 | `0x1` | `0x100` |
| pc | `0x9010f2d70` | `0x9010f2d80` |

## Step 11

- PC: `0x9010f2d90`
- Source line: `119`
- Source text: `const int N   = blockDim.x * gridDim.x * 4;`
- Current instruction: `LDC R7,c[0x0][0x370]`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2d80` | `0x9010f2d90` |

## Step 12

- PC: `0x9010f2da0`
- Source line: `119`
- Source text: `const int N   = blockDim.x * gridDim.x * 4;`
- Current instruction: `MOV R7,R7`

| Register | Old | New |
|---|---|---|
| R7 | `0x0` | `0x2000` |
| pc | `0x9010f2d90` | `0x9010f2da0` |

## Step 13

- PC: `0x9010f2db0`
- Source line: `119`
- Source text: `const int N   = blockDim.x * gridDim.x * 4;`
- Current instruction: `IMAD R6,R6,R7,RZ`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2da0` | `0x9010f2db0` |

## Step 14

- PC: `0x9010f2dc0`
- Source line: `119`
- Source text: `const int N   = blockDim.x * gridDim.x * 4;`
- Current instruction: `IMAD.SHL R6,R6,0x4,RZ`

| Register | Old | New |
|---|---|---|
| R6 | `0x100` | `0x200000` |
| pc | `0x9010f2db0` | `0x9010f2dc0` |

## Step 15

- PC: `0x9010f2dd0`
- Source line: `119`
- Source text: `const int N   = blockDim.x * gridDim.x * 4;`
- Current instruction: `MOV R6,R6`

| Register | Old | New |
|---|---|---|
| R6 | `0x200000` | `0x800000` |
| pc | `0x9010f2dc0` | `0x9010f2dd0` |

## Step 16

- PC: `0x9010f2de0`
- Source line: `None`
- Source text: `None`
- Current instruction: `S2R R7,SR_CTAID.Y`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2dd0` | `0x9010f2de0` |

## Step 17

- PC: `0x9010f2df0`
- Source line: `121`
- Source text: `float *d_Src = d_Input + blockIdx.y * N;`
- Current instruction: `MOV R7,R7`

| Register | Old | New |
|---|---|---|
| R7 | `0x2000` | `0x0` |
| pc | `0x9010f2de0` | `0x9010f2df0` |

## Step 18

- PC: `0x9010f2e00`
- Source line: `121`
- Source text: `float *d_Src = d_Input + blockIdx.y * N;`
- Current instruction: `IMAD R7,R7,R6,RZ`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2df0` | `0x9010f2e00` |

## Step 19

- PC: `0x9010f2e10`
- Source line: `121`
- Source text: `float *d_Src = d_Input + blockIdx.y * N;`
- Current instruction: `MOV R7,R7`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2e00` | `0x9010f2e10` |

## Step 20

- PC: `0x9010f2e20`
- Source line: `121`
- Source text: `float *d_Src = d_Input + blockIdx.y * N;`
- Current instruction: `MOV R7,R7`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2e10` | `0x9010f2e20` |

## Step 21

- PC: `0x9010f2e30`
- Source line: `121`
- Source text: `float *d_Src = d_Input + blockIdx.y * N;`
- Current instruction: `MOV R9,RZ`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2e20` | `0x9010f2e30` |

## Step 22

- PC: `0x9010f2e40`
- Source line: `121`
- Source text: `float *d_Src = d_Input + blockIdx.y * N;`
- Current instruction: `SHF.L.U64.HI R9,R7,0x2,R9`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2e30` | `0x9010f2e40` |

## Step 23

- PC: `0x9010f2e50`
- Source line: `121`
- Source text: `float *d_Src = d_Input + blockIdx.y * N;`
- Current instruction: `SHF.L.U32 R8,R7,0x2,RZ`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2e40` | `0x9010f2e50` |

## Step 24

- PC: `0x9010f2e60`
- Source line: `121`
- Source text: `float *d_Src = d_Input + blockIdx.y * N;`
- Current instruction: `MOV R8,R8`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2e50` | `0x9010f2e60` |

## Step 25

- PC: `0x9010f2e70`
- Source line: `121`
- Source text: `float *d_Src = d_Input + blockIdx.y * N;`
- Current instruction: `MOV R9,R9`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2e60` | `0x9010f2e70` |

## Step 26

- PC: `0x9010f2e80`
- Source line: `121`
- Source text: `float *d_Src = d_Input + blockIdx.y * N;`
- Current instruction: `IADD.64 R2,R2,R8`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2e70` | `0x9010f2e80` |

## Step 27

- PC: `0x9010f2e90`
- Source line: `121`
- Source text: `float *d_Src = d_Input + blockIdx.y * N;`
- Current instruction: `MOV.64 R2,R2`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2e80` | `0x9010f2e90` |

## Step 28

- PC: `0x9010f2ea0`
- Source line: `None`
- Source text: `None`
- Current instruction: `S2R R7,SR_CTAID.Y`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2e90` | `0x9010f2ea0` |

## Step 29

- PC: `0x9010f2eb0`
- Source line: `122`
- Source text: `float *d_Dst = d_Output + blockIdx.y * N;`
- Current instruction: `MOV R7,R7`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2ea0` | `0x9010f2eb0` |

## Step 30

- PC: `0x9010f2ec0`
- Source line: `122`
- Source text: `float *d_Dst = d_Output + blockIdx.y * N;`
- Current instruction: `IMAD R7,R7,R6,RZ`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2eb0` | `0x9010f2ec0` |

## Step 31

- PC: `0x9010f2ed0`
- Source line: `122`
- Source text: `float *d_Dst = d_Output + blockIdx.y * N;`
- Current instruction: `MOV R7,R7`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2ec0` | `0x9010f2ed0` |

## Step 32

- PC: `0x9010f2ee0`
- Source line: `122`
- Source text: `float *d_Dst = d_Output + blockIdx.y * N;`
- Current instruction: `MOV R7,R7`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2ed0` | `0x9010f2ee0` |

## Step 33

- PC: `0x9010f2ef0`
- Source line: `122`
- Source text: `float *d_Dst = d_Output + blockIdx.y * N;`
- Current instruction: `MOV R9,RZ`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2ee0` | `0x9010f2ef0` |

## Step 34

- PC: `0x9010f2f00`
- Source line: `122`
- Source text: `float *d_Dst = d_Output + blockIdx.y * N;`
- Current instruction: `SHF.L.U64.HI R9,R7,0x2,R9`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2ef0` | `0x9010f2f00` |

## Step 35

- PC: `0x9010f2f10`
- Source line: `122`
- Source text: `float *d_Dst = d_Output + blockIdx.y * N;`
- Current instruction: `SHF.L.U32 R8,R7,0x2,RZ`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2f00` | `0x9010f2f10` |

## Step 36

- PC: `0x9010f2f20`
- Source line: `122`
- Source text: `float *d_Dst = d_Output + blockIdx.y * N;`
- Current instruction: `MOV R8,R8`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2f10` | `0x9010f2f20` |

## Step 37

- PC: `0x9010f2f30`
- Source line: `122`
- Source text: `float *d_Dst = d_Output + blockIdx.y * N;`
- Current instruction: `MOV R9,R9`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2f20` | `0x9010f2f30` |

## Step 38

- PC: `0x9010f2f40`
- Source line: `122`
- Source text: `float *d_Dst = d_Output + blockIdx.y * N;`
- Current instruction: `IADD.64 R4,R4,R8`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2f30` | `0x9010f2f40` |

## Step 39

- PC: `0x9010f2f50`
- Source line: `122`
- Source text: `float *d_Dst = d_Output + blockIdx.y * N;`
- Current instruction: `MOV.64 R4,R4`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2f40` | `0x9010f2f50` |

## Step 40

- PC: `0x9010f2f60`
- Source line: `None`
- Source text: `None`
- Current instruction: `IADD R7,R11,-0x1`

| Register | Old | New |
|---|---|---|
| pc | `0x9010f2f50` | `0x9010f2f60` |

