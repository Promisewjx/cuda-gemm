
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <math.h>

#define BLOCK_SIZE 128

// 核函数声明
__global__ void gemm_naive(const float*, const float*, float*, int);
__global__ void gemm_tiled(const float*, const float*, float*, int);
__global__ void gemm_tiled_register_blocking(const float*, const float*, float*, int);
__global__ void gemm_tiled_register_blocking_db(const float*, const float*, float*, int);
__global__ void gemm_tensorcore(const half*, const half*, float*, int);
__global__ void gemm_tensorcore_large_tile(const half*, const half*, float*, int);

float measure_kernel_naive(void (*kernel)(const float*, const float*, float*, int), const float* d_A, const float* d_B, float* d_C, int N, dim3 gridDim, dim3 blockDim) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_C, 0, N * N * sizeof(float));
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

float measure_kernel_tensorcore(void (*kernel)(const half*, const half*, float*, int), const half* d_A, const half* d_B, float* d_C, int N, dim3 gridDim, dim3 blockDim) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_C, 0, N * N * sizeof(float));
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

float measure_kernel_tensorcore_large_tile(void (*kernel)(const half*, const half*, float*, int), 
                                const half* d_A, const half* d_B, float* d_C, int N, 
                                dim3 gridDim, dim3 blockDim, size_t sharedMemSize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_C, 0, N * N * sizeof(float));
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    kernel<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, N);  // 注意这里！
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix size N>\n", argv[0]);
        return -1;
    }

    int N = atoi(argv[1]);
    size_t bytes_float = N * N * sizeof(float);
    size_t bytes_half = N * N * sizeof(half);

    // 分配host内存
    float* h_A_float = (float*)malloc(bytes_float);
    float* h_B_float = (float*)malloc(bytes_float);
    float* h_C = (float*)malloc(bytes_float);

    half* h_A_half = (half*)malloc(bytes_half);
    half* h_B_half = (half*)malloc(bytes_half);

    // 初始化数据
    for (int i = 0; i < N * N; ++i) {
        h_A_float[i] = 1.0f;
        h_B_float[i] = 2.0f;
        h_A_half[i] = __float2half(1.0f);  // 转成half
        h_B_half[i] = __float2half(2.0f);
    }

    // 分配device内存
    float* d_A_float;
    float* d_B_float;
    float* d_C;
    half* d_A_half;
    half* d_B_half;

    cudaMalloc(&d_A_float, bytes_float);
    cudaMalloc(&d_B_float, bytes_float);
    cudaMalloc(&d_C, bytes_float);

    cudaMalloc(&d_A_half, bytes_half);
    cudaMalloc(&d_B_half, bytes_half);

    // 复制数据到device
    cudaMemcpy(d_A_float, h_A_float, bytes_float, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_float, h_B_float, bytes_float, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_half, h_A_half, bytes_half, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_half, h_B_half, bytes_half, cudaMemcpyHostToDevice);

    // kernel配置
    dim3 block_naive(16, 16);
    dim3 grid_naive((N + block_naive.x - 1) / block_naive.x, (N + block_naive.y - 1) / block_naive.y);

    dim3 block_reg(16, 16);
    dim3 grid_reg((N + block_reg.x * 2 - 1) / (block_reg.x * 2), (N + block_reg.y * 2 - 1) / (block_reg.y * 2));

    dim3 block_tensor(8, 8);  // 每warp处理一个16x16 Tile
    dim3 grid_tensor((N + 16 - 1) / 16, (N + 16 - 1) / 16);

    dim3 block_large_tile(256);  // 新大Tile TensorCore
    dim3 grid_large_tile((N + 128 - 1) / 128, (N + 128 - 1) / 128); 
    size_t shared_memory_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(half);  // A_tile + B_tile

    printf("\n===== GEMM Performance Compare (N=%d) =====\n", N);

    float time_naive = measure_kernel_naive(gemm_naive, d_A_float, d_B_float, d_C, N, grid_naive, block_naive);
    double gflops_naive = 2.0 * N * N * N / (time_naive / 1000.0) / 1e9;
    printf("[Naive GEMM] Time = %.3f ms, Performance = %.2f GFLOPS\n", time_naive, gflops_naive);

    float time_tiled = measure_kernel_naive(gemm_tiled, d_A_float, d_B_float, d_C, N, grid_naive, block_naive);
    double gflops_tiled = 2.0 * N * N * N / (time_tiled / 1000.0) / 1e9;
    printf("[Tiled] Time = %.3f ms, Perf = %.2f GFLOPS\n", time_tiled, gflops_tiled);

    float time_reg = measure_kernel_naive(gemm_tiled_register_blocking, d_A_float, d_B_float, d_C, N, grid_reg, block_reg);
    double gflops_reg = 2.0 * N * N * N / (time_reg / 1000.0) / 1e9;
    printf("[Register Blocking GEMM] Time = %.3f ms, Performance = %.2f GFLOPS\n", time_reg, gflops_reg);

    float time_db = measure_kernel_naive(gemm_tiled_register_blocking_db, d_A_float, d_B_float, d_C, N, grid_reg, block_reg);
    double gflops_db = 2.0 * N * N * N / (time_db / 1000.0) / 1e9;
    printf("[Register Blocking + Double Buffering GEMM] Time = %.3f ms, Performance = %.2f GFLOPS\n", time_db, gflops_db);

    float time_tensor = measure_kernel_tensorcore(gemm_tensorcore, d_A_half, d_B_half, d_C, N, grid_tensor, block_tensor);
    double gflops_tensor = 2.0 * N * N * N / (time_tensor / 1000.0) / 1e9;
    printf("[Tensor Core GEMM] Time = %.3f ms, Performance = %.2f GFLOPS\n", time_tensor, gflops_tensor);

    float time_tensor_large = measure_kernel_tensorcore_large_tile(gemm_tensorcore_large_tile, d_A_half, d_B_half, d_C, N, grid_large_tile, block_large_tile, shared_memory_size);
    double gflops_tensor_large = 2.0 * N * N * N / (time_tensor_large / 1000.0) / 1e9;
    printf("[Tensor Core GEMM (large tile)] Time = %.3f ms, Performance = %.2f GFLOPS\n", time_tensor_large, gflops_tensor_large);


    printf("============================================\n");

    // 清理
    cudaFree(d_A_float);
    cudaFree(d_B_float);
    cudaFree(d_C);
    cudaFree(d_A_half);
    cudaFree(d_B_half);

    free(h_A_float);
    free(h_B_float);
    free(h_C);
    free(h_A_half);
    free(h_B_half);

    return 0;
}