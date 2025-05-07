NVCC = nvcc
ARCH = -arch=sm_80
INC = -I./include
SRC = main.cu \
      kernels/gemm_naive.cu \
      kernels/gemm_tiled.cu \
      kernels/gemm_register_blocking.cu \
      kernels/gemm_register_blocking_db.cu \
      kernels/gemm_tensorcore.cu \
      kernels/gemm_tensorcore_large_tile.cu
OUT = gemm_test

all:
	$(NVCC) -O3 $(ARCH) $(INC) $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)
