#pragma once
#include <cuda_fp16.h>  // half 类型

#ifdef __cplusplus
extern "C" {
#endif

// Naive GEMM
__global__ void gemm_naive(const float* A, const float* B, float* C, int N);

// Tiled GEMM
__global__ void gemm_tiled(const float* A, const float* B, float* C, int N);

// Register Blocking
__global__ void gemm_tiled_register_blocking(const float* A, const float* B, float* C, int N);

// Register Blocking + Double Buffering
__global__ void gemm_tiled_register_blocking_db(const float* A, const float* B, float* C, int N);

// Tensor Core
__global__ void gemm_tensorcore(const half* A, const half* B, float* C, int N);

// Tensor Core Large Tile
__global__ void gemm_tensorcore_large_tile(const half* A, const half* B, float* C, int N);

#ifdef __cplusplus
}
#endif
