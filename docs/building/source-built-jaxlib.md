# 从固定源码构建 CPU jaxlib

## 目的

当前环境把 `upstream/jax` 作为 editable Python package 使用，但加载的 C++ runtime 是 PyPI `jaxlib==0.11.1`。JAX 源码版本是 `0.11.2.dev20260830+5832e86644`，所以当前环境适合研究 Python transformation，不能作为 `upstream/xla` C++ 源码已经执行的证据。

本构建闭环需要产生一个由以下源码生成的 CPU wheel：

```text
JAX  5832e866449a41c3eea6333416528039119a0fde
XLA  496bd4bd49db9ecbffd85da630b49c860b724604
Python 3.12
```

构建时必须传入 `--local_xla_path`。这样后续对 `upstream/xla` 的修改才会进入 wheel，而不是由 Bazel 重新取得另一个 external repository。

JAX 固定版本的原始构建说明位于 [`upstream/jax/docs/developer.md`](../../upstream/jax/docs/developer.md)，构建入口是 [`upstream/jax/build/build.py`](../../upstream/jax/build/build.py)。本页只记录本分析仓库的额外 provenance、安装和回滚约束。

## 当前前置检查

从仓库根目录运行：

```bash
.venv/bin/python tools/check-jaxlib-build-env.py
```

严格模式会在缺少本地 C/C++ compiler 或源码目录异常时返回非零：

```bash
.venv/bin/python tools/check-jaxlib-build-env.py --strict
```

2026-09-07 的当前检查结果：

- Python 3.12.3 可用；
- JAX 要求 Bazel 8.7.0；
- 本地尚无 `bazel`/`bazelisk`；`build.py` 可以下载匹配 Bazel，但这需要网络；
- Clang 18.1.3 已安装，`clang++` 位于 `/usr/bin/clang++`；
- 工作区约有 936 GiB 可用空间；
- 主机约有 15 GiB RAM、4 GiB swap 和 12 个逻辑 CPU。

首次下载 Bazel 后重新运行 preflight，把新结果保存到构建 fingerprint。

## 配置检查

正式长时间构建前，先让 JAX 输出将要执行的 Bazel 命令：

```bash
cd upstream/jax
../../.venv/bin/python build/build.py build \
  --wheels=jaxlib \
  --python_version=3.12 \
  --local_xla_path=../xla \
  --output_path=/tmp/jax-source-analysis-wheel \
  --dry_run \
  --verbose
```

需要确认输出至少满足：

1. wheel target 是 `//jaxlib/tools:jaxlib_wheel`；
2. hermetic Python 是 3.12；
3. XLA override 指向本仓库的 `upstream/xla`；
4. 没有启用 CUDA、ROCm 或 OneAPI plugin；
5. 输出路径不覆盖当前 `.venv`。

## 首个基线构建

预期命令如下；实际执行前应根据 preflight 选择 compiler，并把完整命令写入 fingerprint：

```bash
cd upstream/jax
../../.venv/bin/python build/build.py build \
  --wheels=jaxlib \
  --python_version=3.12 \
  --local_xla_path=../xla \
  --output_path=/tmp/jax-source-analysis-wheel \
  --verbose \
  --detailed_timestamped_log
```

如果受 15 GiB 内存限制，需要降低 Bazel 并行度时，通过重复的 `--bazel_options` 传入经过本机验证的资源设置，并把设置写入 fingerprint。不要只在交互 shell 中设置未记录的构建选项。

## Wheel fingerprint

构建结束后至少记录：

```text
JAX commit
XLA commit
StableHLO/Shardy/LLVM commits
dirty state of JAX and XLA
Python and build.py command
Bazel version
C/C++ compiler path and version
all Bazel startup/build options
wheel filename, size and SHA-256
jaxlib version and embedded git hash
shared-library dependencies
build start/end time and result
```

fingerprint 写入 `manifests/build-fingerprints/`。wheel 和 Bazel cache 不提交到 Git；只保存 hash、日志摘要和重建命令。

## 隔离安装与验证

不要直接覆盖项目当前 `.venv`。首个 source-built wheel 使用临时 uv environment 或独立 venv 验证。验证程序至少输出并断言：

- `jax.__file__` 仍指向固定的 editable `upstream/jax`；
- `jaxlib.__file__` 指向临时环境中新安装的 wheel；
- JAX/jaxlib compatibility check 通过；
- 仅发现预期 CPU devices；
- Lab 001 的 Jaxpr、StableHLO、数值和 cache probe 通过；
- 一个直接触达新增 C++ 诊断标记的 probe 通过。

只有以上验证全部通过，才能把该 wheel 标记为 `RUN-CPU` 基线。

## 证明本地 XLA 修改进入运行时

基线 wheel 成功后增加一个默认关闭、环境变量控制的无语义诊断点。诊断点应位于后续 pass 实验会经过的 XLA 路径，并具备以下性质：

1. 未设置环境变量时没有输出或行为变化；
2. patch 可以用 `git apply --check` 检查并反向应用；
3. 重新构建后，probe 能观察唯一版本字符串或诊断事件；
4. 使用原始 PyPI wheel 时 probe 不会误报；
5. 移除 patch 并重建后恢复基线。

这个步骤证明“阅读的 XLA 工作树”“构建输入”和“实际加载的二进制”是同一个版本闭包。

## 安装、缓存与回滚

每次切换 wheel 时同时处理以下状态：

- Python environment 中的 jaxlib package；
- JAX in-memory compilation cache；
- persistent compilation cache；
- Bazel build cache；
- 已注册的 HLO transformations；
- 当前 shell 中的 `JAX_*`/`XLA_FLAGS` 配置。

回滚完成的判据是：

1. `jaxlib` 路径、版本和 hash 回到记录的基线；
2. JAX/JAXLIB import 与 Lab 001 通过；
3. JAX 和 XLA submodule 的工作树状态符合 fingerprint；
4. 诊断标记不再出现。

## 尚未执行

本文目前是构建设计，证据标签为 `SOURCE-ONLY`。compiler 和 Bazel 尚未安装，source-built wheel 尚未生成。完成首次构建后，应把真实命令、耗时、资源设置、wheel fingerprint 和验证结果替换进本页，并更新 `PLAN.md` 的 P1 状态。
