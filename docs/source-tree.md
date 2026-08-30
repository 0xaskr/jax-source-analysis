# 源码树、依赖关系与边界

## 判断“上游”的标准

这里的方向是“硬件为上游、开发者为下游”。一个项目满足下列任一条件时才进入源码树：

1. JAX 的 Python 包直接导入或声明它；
2. JAX/jaxlib 的 Bazel 构建直接声明它；
3. JAX 锁定的 XLA 继续通过 Bzlmod、Bazel repository rule 或源码构建声明它；
4. 它实现 JAX 到硬件所必需的编译器、PJRT、kernel 或系统运行时接口。

反过来，导入 JAX、在 JAX 之上提供模型/算子/应用的仓库是下游。Tokamax 即使与 Pallas kernel 有关，依赖方向仍然是 `Tokamax → JAX`，所以不在本树中。

## 主干依赖链

```text
开发者程序
  ↓
JAX API / transformations / jaxpr
  ↓
JAX MLIR lowering + jaxlib bindings
  ↓
StableHLO ── Shardy
  ↓
XLA HLO optimizations / code generation
  ├── CPU backend → LLVM → oneDNN/Eigen/XNNPACK/OpenMP → CPU
  ├── GPU backend → LLVM/Triton → CUDA/ROCm/SYCL libraries → GPU
  └── PJRT C API → libtpu.so（闭源边界）→ TPU
```

JAX 对 XLA 是直接源码依赖；StableHLO、Shardy、LLVM、Triton 和各类 kernel/runtime 项目主要是 `JAX → XLA → dependency` 的递归依赖。并非所有条件分支会在同一次构建中启用，例如 NVIDIA、AMD 和 Intel GPU 依赖互斥或按配置选择。

## 目录与依赖归属

| 目录 | 内容 | 依赖性质 |
|---|---|---|
| `upstream/jax` | JAX Python、jaxlib、Pallas/Mosaic | 分析边界与直接源码 |
| `upstream/xla` | HLO、编译管线、PJRT、CPU/GPU 后端 | JAX 直接锁定 |
| `upstream/stablehlo` | StableHLO dialect 与 passes | 经 XLA 递归依赖 |
| `upstream/shardy` | sharding dialect/passes | 经 XLA 递归依赖 |
| `upstream/llvm-project` | LLVM、MLIR、NVPTX/AMDGPU 等后端 | 经 XLA 递归依赖 |
| `upstream/triton` | GPU kernel dialect、passes、codegen | 经 XLA 递归依赖 |
| `upstream/cpu` | oneDNN、Eigen、XNNPACK、OpenMP 等 | XLA CPU 条件依赖 |
| `upstream/gpu` | CUTLASS、NCCL、ROCm LLVM、Level Zero 等 | XLA GPU 条件依赖 |
| `upstream/runtime` | DLPack、Gloo、oneCCL、nanobind 等 | XLA/JAX 公共运行时 |
| `upstream/python` | NumPy、SciPy、ml-dtypes、opt_einsum | JAX Python 直接依赖 |
| `upstream/tooling` | protobuf、gRPC、Bazel rules、XProf 等 | 构建、绑定、RPC 或可选分析 |

精确来源见 `upstream-sources.lock`。每行包含分类、放置路径、官方 Git URL、JAX/XLA 所需 ref、声明该依赖的源码位置、可选的稀疏检出目录和 checkout 策略。`lazy` 仍是标准 submodule 和精确 gitlink，只是工作树暂未下载；它不等于遗漏依赖。

## Bazel include 与真实源码

例如 XLA 在 `upstream/xla/third_party/llvm_openmp/workspace.bzl` 下载 OpenMP 10.0.1，并用同目录的 BUILD overlay 构建它。分析树把对应官方 Git tag 放在 `upstream/cpu/llvm-project-10.0.1/openmp/`：

```text
upstream/xla/third_party/llvm_openmp/       XLA 的下载规则、patch、BUILD overlay
upstream/cpu/llvm-project-10.0.1/openmp/    被下载和编译的原始上游源码
```

其他 Bazel archive 依赖采用相同原则：overlay 留在 XLA，原始仓库作为外层 submodule 放到 CPU/GPU/runtime/tooling 的合理位置。

## 无法获得开源实现的边界

- TPU：JAX 会动态加载 `libtpu.so`，其 PJRT 接口及 JAX 的 Mosaic TPU dialect 是可读的，但 `libtpu` 内部 TPU 编译器、运行时和驱动实现没有对应公开源码仓库。
- NVIDIA：LLVM NVPTX、Triton、CUTLASS、NCCL、NVSHMEM、cuDNN frontend 等开源部分已收录；CUDA driver、cuBLAS、cuDNN backend 等二进制实现不是开源源码。
- 系统库：个别配置引用宿主系统已经安装的库，XLA 不锁定其源码版本；这种依赖在文档中标明，但不会伪造一个“精确递归依赖”的任意 Git commit。
