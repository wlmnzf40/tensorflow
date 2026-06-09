# TF Serving GEMM Benchmark

纯 C/C++ 形态的 TensorFlow Serving + GEMM 算子 benchmark 示例。

## 整体流程

```
generate_model.py  →  gemm_model/1/   (SavedModel)
start_serving.sh   →  TF Serving (Docker)   :8500 gRPC  :8501 REST
benchmark_rest.py  →  Python REST 客户端  (只需 requests + numpy)
benchmark_grpc.py  →  Python gRPC 客户端  (需 tensorflow-serving-api)
benchmark.cc       →  C++ REST 客户端    (需 libcurl)
```

## 1. 生成模型

```bash
# 方式 A：本地 TF Python 可用时
python3 generate_model.py

# 方式 B：本地 TF 有符号缺失等问题时（推荐）
python3 generate_model.py --docker
```

生成后应有 `gemm_model/1/saved_model.pb`。

## 2. 启动 TF Serving

```bash
chmod +x start_serving.sh stop_serving.sh
./start_serving.sh

# 验证服务
curl http://localhost:8501/v1/models/gemm
```

## 3. 运行 benchmark

### Python REST 客户端（最简单，无 TF 依赖）

```bash
pip install requests numpy
python3 python_client/benchmark_rest.py --M 1024 --K 1024 --N 1024 --iters 50
```

### Python gRPC 客户端

```bash
pip install grpcio tensorflow-serving-api numpy
python3 python_client/benchmark_grpc.py --M 1024 --K 1024 --N 1024
```

### C++ REST 客户端

```bash
sudo apt-get install -y libcurl4-openssl-dev cmake

cd cpp_client
mkdir build && cd build
cmake ..
make -j$(nproc)

./gemm_benchmark --M 1024 --K 1024 --N 1024 --iters 50 --warmup 10
```

## 参数说明

| 参数       | 含义                      | 默认值 |
|------------|---------------------------|--------|
| `--M`      | A 矩阵行数                | 512    |
| `--K`      | A 列 / B 行               | 512    |
| `--N`      | B 矩阵列数                | 512    |
| `--iters`  | 正式测量轮数              | 50     |
| `--warmup` | 预热轮数（不计入统计）     | 10     |
| `--host`   | TF Serving 主机地址       | localhost |
| `--port`   | 端口 (REST=8501, gRPC=8500)| 8501  |

## 典型输出

```
==================================================
  Matrix sizes : (1024x1024) x (1024x1024)
  FLOPs/call   : 2.147 GFLOP
  Latency avg  : 12.34 ms
  Latency p50  : 12.10 ms
  Latency p99  : 15.20 ms
  Throughput   : 174.02 GFLOPS
==================================================
```

## 关闭服务

```bash
./stop_serving.sh
```
