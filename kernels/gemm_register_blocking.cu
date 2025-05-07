#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void gemm_tiled_register_blocking(const float* A, const float* B, float* C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    // 每个 thread 处理 2x2 个元素
    int row = blockIdx.y * TILE_SIZE + threadIdx.y * 2;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x * 2;

    // 4个寄存器，保存2x2子块的累加结果
    float regC00 = 0.0f, regC01 = 0.0f;
    float regC10 = 0.0f, regC11 = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 搬A块
        if (row + 0 < N && t * TILE_SIZE + threadIdx.x < N)
            tile_A[threadIdx.y * 2 + 0][threadIdx.x] = A[(row + 0) * N + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y * 2 + 0][threadIdx.x] = 0.0f;

        if (row + 1 < N && t * TILE_SIZE + threadIdx.x < N)
            tile_A[threadIdx.y * 2 + 1][threadIdx.x] = A[(row + 1) * N + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y * 2 + 1][threadIdx.x] = 0.0f;

        // 搬B块
        if (col + 0 < N && t * TILE_SIZE + threadIdx.y < N)
            tile_B[threadIdx.y][threadIdx.x * 2 + 0] = B[(t * TILE_SIZE + threadIdx.y) * N + col + 0];
        else
            tile_B[threadIdx.y][threadIdx.x * 2 + 0] = 0.0f;

        if (col + 1 < N && t * TILE_SIZE + threadIdx.y < N)
            tile_B[threadIdx.y][threadIdx.x * 2 + 1] = B[(t * TILE_SIZE + threadIdx.y) * N + col + 1];
        else
            tile_B[threadIdx.y][threadIdx.x * 2 + 1] = 0.0f;

        __syncthreads();

        // 累加
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float regA0 = tile_A[threadIdx.y * 2 + 0][k];
            float regA1 = tile_A[threadIdx.y * 2 + 1][k];
            float regB0 = tile_B[k][threadIdx.x * 2 + 0];
            float regB1 = tile_B[k][threadIdx.x * 2 + 1];

            regC00 += regA0 * regB0;
            regC01 += regA0 * regB1;
            regC10 += regA1 * regB0;
            regC11 += regA1 * regB1;
        }

        __syncthreads();
    }

    // 写回C
    if (row + 0 < N && col + 0 < N)
        C[(row + 0) * N + col + 0] = regC00;
    if (row + 0 < N && col + 1 < N)
        C[(row + 0) * N + col + 1] = regC01;
    if (row + 1 < N && col + 0 < N)
        C[(row + 1) * N + col + 0] = regC10;
    if (row + 1 < N && col + 1 < N)
        C[(row + 1) * N + col + 1] = regC11;
}