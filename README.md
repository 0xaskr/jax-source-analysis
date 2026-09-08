# JAX 软件栈：源码分析与 Hack 实践

本仓库的目标是对 JAX 的 CPU、GPU、TPU 软件栈进行系统性分析，从 JAX 出发一路向硬件方向探索；同时 hack 其中一部分，按需手动编译修改后的组件，并验证修改后的实现。当前源码与实验基线固定为：

- JAX `5832e866449a41c3eea6333416528039119a0fde`（源码版本
  `0.11.2.dev20260830+5832e86644`）
- JAX 在 `MODULE.bazel` 中锁定的 XLA `496bd4bd49db9ecbffd85da630b49c860b724604`

边界以 `upstream/jax` 为准：JAX 使用、构建或运行时所依赖的项目属于这里的“上游”；
下游框架内部不分析。Tokamax 和自研框架接入后只保留固定 revision、可复现的纯
JAX/Pallas 入口和调用契约，用于度量 JAX→TPU 路径覆盖。

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

当前共有 102 个已登记的 submodule，其中 85 个采用 eager 策略，17 个体积较大或只服务于构建、兼容性的项目采用 lazy 策略。登记状态不表示当前机器已经下载源码；环境同步入口先恢复 baseline 必需的五个核心源码树。唯一的非-submodule 源码例外是 `upstream/gpu/nvshmem-3.1.7-source/`：XLA 锁定的 NVIDIA 3.1.7 发布包没有对应的公开 Git ref，因此保留官方源码包、SHA-256 和来源标记；同时另有 `upstream/gpu/nvshmem/` submodule 用于阅读公开 Git 源码。

## 初始化与校验

Linux x86_64 新机器先执行项目同步入口：

```bash
python3 -B tools/sync-environment.py sync
python3 -B tools/sync-environment.py check
```

入口从 `upstream-sources.lock` 和已提交的 gitlink 直接获取 JAX、XLA、StableHLO、Shardy、LLVM 的固定 commit；核对 uv/Bazel 指纹，执行 `uv sync --locked`，最后校验实际 CPU baseline、历史证据包和恢复状态，并运行当前 Lab 001 的 CPU 断言。已有源码改动、非空的未初始化目录、错误下载指纹或任何命令失败都会停止同步；脚本不暂存文件。可再生的 ignored `.pyc/.pyo` 会移入 `artifacts/environment/bytecode/` 备份。

工具、容器镜像 digest、Ubuntu 软件包快照及 Python 发行版修订固定在 [`env/environment.lock.json`](env/environment.lock.json)。宿主机 CPU 环境要求 Python 3.12.3；严格构建环境使用 Docker：

```bash
python3 -B tools/sync-environment.py docker-build
python3 -B tools/sync-environment.py docker
```

容器入口执行同一套同步和验证，再运行严格 jaxlib preflight。它以调用者 UID/GID 运行，挂载当前仓库，并把 `artifacts/environment/docker-venv/` 映射为容器内的 `.venv`，与宿主机 venv 分开。可在 `docker --` 后传入要运行的命令；每次运行使用与当前构建配置匹配的本地 image ID。所有环境缓存都位于 ignored 的 `artifacts/environment/` 下。完整的工具链边界和更新流程见[构建说明](docs/building/source-built-jaxlib.md#跨机器环境同步)。

需要展开其余 eager 源码及精确 GPU archive 时，再使用原有全量入口：

```bash
bash tools/fetch-upstream-sources.sh
bash tools/fetch-source-archives.sh
bash tools/verify-upstream-sources.sh
```

第一个脚本按锁文件初始化并校准主路径 submodule，同时跳过 lazy 项，并会暂存 `.gitmodules` 和 gitlink；使用前先处理现有暂存内容。OpenMP 使用 blobless 浅克隆，只展开 LLVM 10.0.1 的 `openmp/`。第二个脚本下载并校验唯一的精确源码包。第三个脚本检查全量清单、`.gitmodules`、gitlink、实际 HEAD、lazy 状态和 archive 标记是否一致，不能用它代替五个核心源码的恢复门禁。

## Python 分析环境

仓库使用 uv 管理可复现的 CPU 分析环境。Python 固定为 3.12.3，JAX 以 editable 方式直接使用 `upstream/jax` 中的源码，CPU `jaxlib` 固定为 0.11.1；其余传递依赖的精确版本和文件哈希记录在 `uv.lock`。

```bash
python3 -B tools/sync-environment.py sync
.venv/bin/python -B -c 'import jax; print(jax.__version__, jax.devices())'
```

JupyterLab 作为可选的 uv 依赖组维护。运行第一个交互式源码分析 Lab：

```bash
python3 -B tools/sync-environment.py sync --notebook
PYTHONDONTWRITEBYTECODE=1 .venv/bin/jupyter lab labs/001-jit-cpu/jupyter.ipynb
```

正常使用时不要删除或绕过 `uv.lock`。只有在有意更新分析基线时才重新解析依赖，并同时审查锁文件变化。

需要阅读某个 lazy 项时，显式传入路径即可，例如：

```bash
bash tools/fetch-upstream-sources.sh upstream/runtime/boringssl
# 或使用标准 Git 命令：
git submodule update --init --depth 1 -- upstream/runtime/boringssl
```

继续阅读：

- [长期执行计划](PLAN.md)
- [当前机器可读状态](manifests/status.json)（运行
  `.venv/bin/python tools/project-status.py` 查看，使用 `--check` 执行恢复门禁）
- [机器可读覆盖清单](manifests/coverage.json)（`covered` 必须与其中的实际 depth 一起解读）
- [JAX → TPU 全栈架构图](docs/architecture/00-whole-stack.md)
- [IFRT/PJRT 与 TPU runtime 控制面](docs/architecture/03-runtime-control-plane.md)
- [JAX → TPU 术语表](docs/index/glossary.md)
- [源码分析证据约定](docs/contributing/evidence-conventions.md)
- [从固定源码构建 jaxlib](docs/building/source-built-jaxlib.md)
- [源码树、依赖关系与边界](docs/source-tree.md)
- [CPU、GPU、TPU 阅读路径](docs/reading-paths.md)
- [Lab 001：追踪 `jax.jit` 的 CPU 执行路径](labs/001-jit-cpu/README.md)（Marimo 实验台 + CodeTour 源码走读）
