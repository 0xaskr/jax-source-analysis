# 固定源码的 CPU jaxlib 构建

对应 kickoff R04，以及消除 R02 native `VERSION-SKEW` 的前置步骤。

## 已完成的环境核查

宿主机预检报告 Python executable 字节不匹配、缺少固定 Clang/Clang++。
固定 Docker 镜像 `sha256:abad11bf00a9611382e946ab243d7e3d105a4019ba3cf6bd542223debb18d6e8`
的配方标签与 `env/` 一致；在关闭网络、只读挂载仓库、以当前用户运行的容器内，
`tools/check-jaxlib-build-env.py --json` 返回 `ok=true`、无 blocker。
原始报告保存在 `artifacts/jax-stack/build-preflight-001/`。

## 构建方式

使用已存在并经过源码阅读的 `tools/build-jaxlib.py` 记录命令、attempt、日志与 wheel。
它调用上游 `build/build.py build --wheels=jaxlib --python_version=3.12`，并使用固定
XLA、Bazel 8.7.0、Clang 18.1.3。工作区另建 JAX/XLA 的同 revision 克隆；构建容器
把它们挂到工具要求的逻辑路径，原始五个上游 checkout 以只读方式保护。

- 镜像和宿主机原始仓库只读；仅隔离克隆、构建产物与 build manifest 目录可写。
- 原有宿主机 `.venv` 不被覆盖；构建使用保留的 Docker venv。
- Bazel 使用 2 个 jobs；容器限 4 CPU、24 GiB 内存，资源配置保存到启动记录。
- 构建 ID：`kickoff-cpu-source-001`。实际容器 ID 和挂载映射保存到
  `artifacts/jax-stack/source-build-001/launch.json`。

构建是否运行必须通过该容器的当前 `docker inspect` 状态判断。wrapper 的 PID
属于容器 PID namespace，不能在宿主机凭同号 PID 或一个旧 manifest 判定它已停止，
更不能因此启动重复构建。

## 仍需验收

1. 构建真实结束且 wheel/RECORD/native payload 验证通过。
2. 在隔离环境安装 wheel，核对实际加载二进制及 build revision，重新执行 matmul。
3. 记录 Bazel 外部 module/repository 来源与下载完整性。现有 wrapper 固定
   `--lockfile_mode=off`；在外部依赖闭包补齐前，不宣称整个构建可完整离线重放。
4. 做一个有预期日志变化的可逆诊断修改，重新编译、加载、对照并回滚。

容器预检通过不等于构建成功；wheel 构建成功不等于运行验证或 Hack 验收完成。

## 当前 attempt

容器 `jax-kickoff-cpu-source-001` 已启动，ID 为
`5cd1c2485f1b6960a9ddcb522b145d343fb4b61047466de2320b96843b2680c3`。
实际状态记录在忽略目录的 `container-state.json`；这是带时效的快照，恢复时必须重新 inspect。
2026-09-14 本轮观察已进入 C++ 编译（LLVM/MLIR 等），尚无成功 wheel/加载验收。
日志：`artifacts/builds/kickoff-cpu-source-001/attempts/0001/build.log`。

JAX 隔离树由 shared clone 创建。XLA 的 shared clone 因 partial clone 中无关历史对象缺失
而失败；随后用新 Git 仓库、只读借用原 objects、保留 shallow 边界并 checkout 当前固定
revision，成功得到干净隔离树。没有修改或修复原源码，也没有将缺失历史误判为当前树损坏。

构建驱动使用预检通过的 Python 3.12.3；Bazel 日志还下载了 hermetic Python 3.12.13，
二者职责不同。后续要记录实际外部依赖闭包（包括 hermetic Python、LLVM/MLIR 来源），
不能仅凭驱动环境固定就声称所有构建输入都与五个本地 checkout 一致。
