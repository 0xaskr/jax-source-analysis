# JAX 上游源码分析树

这个仓库用于从 JAX 源码一路向硬件方向阅读 CPU、GPU、TPU 软件栈。基线固定为：

- JAX `5832e866449a41c3eea6333416528039119a0fde`（源码版本 `0.11.2`）
- JAX 在 `MODULE.bazel` 中锁定的 XLA `496bd4bd49db9ecbffd85da630b49c860b724604`

边界以 `upstream/jax` 为准：JAX 使用、构建或运行时所依赖的项目属于这里的“上游”；使用 JAX 构建应用或库的项目属于下游，不收录。Tokamax、jax-triton、jax-tpu-embedding 等下游项目已明确排除。

## 源码树

```text
upstream/
├── jax/                 JAX 与 jaxlib
├── xla/                 编译器、PJRT、CPU/GPU 后端和 StreamExecutor
├── stablehlo/           可移植 HLO 方言
├── shardy/              分片传播与 SPMD 表达
├── llvm-project/        XLA 锁定的 LLVM/MLIR
├── triton/              XLA GPU kernel 编译路径
├── cpu/                 CPU kernel、数学库、线程与体系结构依赖
├── gpu/                 CUDA/ROCm/SYCL 的开源编译与运行时组件
├── runtime/             跨硬件通信、序列化和基础运行时
├── python/              JAX 的 Python 数值依赖
└── tooling/             Bazel、绑定、RPC、性能分析等构建依赖
```

每个目录中的项目保留自己的原始源码结构。不会把依赖仓库嵌入 `upstream/xla/third_party/`，因为该目录已经属于 XLA submodule；外层仓库无法在一个 gitlink 内安全追踪另一个 gitlink。XLA 的 Bazel overlay 仍在原位置，真实依赖源码按用途放在相邻的源码树中，并由 `upstream-sources.lock` 的 `pinned_by` 字段建立对应关系。

当前共有 102 个已登记的 submodule。其中 85 个主干/硬件路径已经初始化；17 个体积较大或只服务于构建、兼容性的项目采用 lazy 策略，只登记官方 URL 与精确 gitlink，暂不下载工作树。唯一的非-submodule 源码例外是 `upstream/gpu/nvshmem-3.1.7-source/`：XLA 锁定的 NVIDIA 3.1.7 发布包没有对应的公开 Git ref，因此保留官方源码包、SHA-256 和来源标记；同时另有 `upstream/gpu/nvshmem/` submodule 用于阅读公开 Git 源码。

## 初始化与校验

在这个分析仓库中执行：

```bash
bash tools/fetch-upstream-sources.sh
bash tools/fetch-source-archives.sh
bash tools/verify-upstream-sources.sh
```

第一个脚本按锁文件初始化并校准主路径 submodule，同时跳过 lazy 项；OpenMP 使用 blobless 浅克隆，只展开 LLVM 10.0.1 的 `openmp/`。第二个脚本下载并校验唯一的精确源码包。第三个脚本检查清单、`.gitmodules`、gitlink、实际 HEAD、lazy 状态和 archive 标记是否一致。

## Python 分析环境

仓库使用 uv 管理可复现的 CPU 分析环境。Python 固定为 3.12.3，JAX 以 editable 方式直接使用 `upstream/jax` 中的源码，CPU `jaxlib` 固定为 0.11.1；其余传递依赖的精确版本和文件哈希记录在 `uv.lock`。

```bash
uv sync --locked
uv run python -c 'import jax; print(jax.__version__, jax.devices())'
```

正常使用时不要删除或绕过 `uv.lock`。只有在有意更新分析基线时才重新解析依赖，并同时审查锁文件变化。

需要阅读某个 lazy 项时，显式传入路径即可，例如：

```bash
bash tools/fetch-upstream-sources.sh upstream/runtime/boringssl
# 或使用标准 Git 命令：
git submodule update --init --depth 1 -- upstream/runtime/boringssl
```

继续阅读：

- [源码树、依赖关系与边界](docs/source-tree.md)
- [CPU、GPU、TPU 阅读路径](docs/reading-paths.md)
