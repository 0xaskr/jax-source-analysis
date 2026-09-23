# JAX source workspace

保留固定版本的上游源码、开发环境，以及可复用的工具和实验源码。

## 源码与环境

- `upstream/`：JAX、XLA、StableHLO、Shardy、LLVM/MLIR 及登记的依赖源码。
- `upstream-sources.lock`、`source-archives.lock`、`.gitmodules`：源码版本和来源。
- `pyproject.toml`、`uv.lock`、`.python-version`：Python 环境定义。
- `env/`、`manifests/baseline.json`：固定环境、工具链和运行二进制基线。
- `tools/`、`manifests/schema/`：同步、构建、采集和校验工具源码及数据格式定义。
- `labs/`：保留的 Python、Notebook 源码和可逆补丁。
- `research`: 对于jax软件栈的探索产物。

## 环境入口

```bash
# 检查已有源码、环境和 CPU probe。
python3 -B tools/sync-environment.py check

# 恢复固定环境，包含可选 Notebook 依赖。
python3 -B tools/sync-environment.py sync --notebook

# 固定系统工具链容器。
python3 -B tools/sync-environment.py docker-build
python3 -B tools/sync-environment.py docker
```

源码与运行二进制是否匹配以实际校验为准；CPU 验证不能作为 TPU 执行证据。

## 当前研究

按 Outline kickoff revision 51 开展的新研究见
[研究索引](research/index.md)分为软件栈开放探索与固定调用到 LLO 的追踪，包含源码材料、CPU 产物和 Notebook。
