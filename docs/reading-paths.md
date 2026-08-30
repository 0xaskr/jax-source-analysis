# CPU、GPU、TPU 阅读路径

## 共同主干

建议先用一个简单的 `jax.jit` 函数贯穿这些位置：

1. `upstream/jax/jax/_src/api.py`：用户 API 与 transformation 入口；
2. `upstream/jax/jax/_src/core.py`、`interpreters/partial_eval.py`：tracing、jaxpr 与抽象求值；
3. `upstream/jax/jax/_src/interpreters/mlir.py`：jaxpr 到 MLIR/StableHLO 的 lowering；
4. `upstream/jax/jax/_src/compiler.py`：编译选项、lower/compile 调用；
5. `upstream/jax/jax/_src/xla_bridge.py`：PJRT backend 发现和 client 创建；
6. `upstream/jax/jaxlib/`：Python/C++ 边界、FFI 与设备对象；
7. `upstream/stablehlo/stablehlo/`：StableHLO dialect、验证和 passes；
8. `upstream/xla/xla/`：HLO 优化、后端 codegen 与 PJRT runtime。

## CPU 路径

```text
JAX → StableHLO → XLA HLO → XLA CPU backend → LLVM machine code → CPU runtime
```

重点源码：

- `upstream/xla/xla/backends/cpu/`：新 CPU backend、codegen、runtime、collectives；
- `upstream/xla/xla/service/cpu/`：CPU HLO passes 与 oneDNN integrations；
- `upstream/xla/xla/pjrt/cpu/`：CPU PJRT client；
- `upstream/llvm-project/llvm/` 与 `mlir/`：中间表示、优化和目标机器码；
- `upstream/cpu/onednn-*`、`eigen/`、`xnnpack/`、`compute-library/`：高性能算子与架构 kernel；
- `upstream/cpu/llvm-project-10.0.1/openmp/runtime/`：oneDNN OpenMP 配置使用的 `libiomp5` 线程运行时。

OpenMP 不是 JAX 语义层；它是某些 CPU 优化 kernel 的条件递归运行时。XLA 自身还有独立线程池路径，不应把所有 CPU 并行都归因于 OpenMP。

## GPU 路径

```text
JAX → StableHLO → XLA GPU → LLVM/Triton kernel codegen
    → StreamExecutor/PJRT → CUDA、ROCm 或 SYCL runtime → GPU
```

共同入口：

- `upstream/xla/xla/backends/gpu/` 与 `xla/service/gpu/`：GPU 优化、fusion、emitter、runtime；
- `upstream/xla/xla/stream_executor/{cuda,rocm,sycl}/`：驱动和库调用抽象；
- `upstream/xla/xla/pjrt/gpu/`：GPU PJRT；
- `upstream/triton/`：Triton dialect、passes 与 GPU codegen；
- `upstream/jax/jax/_src/pallas/{triton,mosaic_gpu}/` 与 `jaxlib/mosaic/gpu/`：JAX kernel DSL 的 GPU 路径。

按厂商继续：

- NVIDIA：`cutlass-*`、`cudnn-frontend`、`nccl`、`nvshmem*`、`cccl`、`cuda-tile`；闭源 CUDA/cuBLAS/cuDNN backend 止于其公开 API；
- AMD：`rocm-llvm-project`、`roc-mori`、`rdma-core`、`libdrm`，并从 XLA 的 `stream_executor/rocm` 阅读宿主 ROCm 库接口；
- Intel：`intel-xpu-triton`、`level-zero`，以及 XLA 的 SYCL StreamExecutor 路径。

## TPU 路径

```text
JAX → StableHLO/Shardy → PJRT C API → libtpu.so → TPU
          └→ Pallas/Mosaic TPU dialect 与 lowering
```

可读源码：

- `upstream/jax/jax/_src/pallas/mosaic/`：Pallas 的 TPU/Mosaic Python lowering；
- `upstream/jax/jaxlib/mosaic/dialect/tpu/`：TPU MLIR dialect 定义；
- `upstream/jax/jaxlib/mosaic/python/tpu.py`：Python binding；
- `upstream/jax/jax/_src/xla_bridge.py`：`libtpu.so` 的动态加载与 PJRT 初始化；
- `upstream/xla/xla/pjrt/c/`：PJRT C API；
- `upstream/xla/xla/pjrt/tpu/`：公开的 TPU compiler variant 接口；
- `upstream/shardy/shardy/`：跨设备 sharding 表达与传播。

到 `libtpu.so` 后就是明确的源码边界。没有把 TPU embedding、Tokamax 等 JAX 上层项目放进来，因为它们不能补上这个闭源缺口，而且依赖方向是下游。
