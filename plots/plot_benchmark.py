import pandas as pd
import matplotlib.pyplot as plt

# 整理数据
data = {
    "Version": [
        "Naive", 
        "Tiled", 
        "Register Blocking", 
        "Register Blocking + DB", 
        "Tensor Core", 
        "Tensor Core Large Tile"
    ],
    "Time (ms)": [
        69.714, 
        34.323, 
        16.659, 
        13.621, 
        0.559, 
        0.081
    ],
    "GFLOPS": [
        1971.47, 
        4004.23, 
        8249.91, 
        10090.04, 
        245820.00, 
        1698958.63
    ]
}

# 转成 DataFrame
df = pd.DataFrame(data)

# 可视化
plt.figure(figsize=(10, 6))
plt.bar(df["Version"], df["GFLOPS"], color='royalblue')
plt.ylabel("Performance (GFLOPS)")
plt.title("GEMM Kernel Performance Comparison (N=4096)")
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存并显示图像
plt.tight_layout()
plt.savefig("benchmark_plot.png")
plt.show()
