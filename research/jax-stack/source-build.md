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
- Bazel 使用 4 个 jobs，JVM 上限 4 GiB；容器限 4 CPU、24 GiB 内存，资源配置保存到启动记录。
- 当前构建 ID：`kickoff-cpu-source-002`。实际容器 ID 和挂载映射保存到
  `artifacts/jax-stack/source-build-002/launch.json`。

构建是否运行必须通过该容器的当前 `docker inspect` 状态判断。wrapper 的 PID
属于容器 PID namespace，不能在宿主机凭同号 PID 或一个旧 manifest 判定它已停止，
更不能因此启动重复构建。

## 仍需验收

1. 构建真实结束且 wheel/RECORD/native payload 验证通过。
2. 在隔离环境安装 wheel，核对实际加载二进制及 build revision，重新执行 matmul。
3. 记录 Bazel 外部 module/repository 来源与下载完整性。现有 wrapper 固定
   `--lockfile_mode=off`；在外部依赖闭包补齐前，不宣称整个构建可完整离线重放。
4. 做一个有预期日志变化的可逆诊断修改，重新编译、加载、对照并回滚。

[独立运行环境](runtime-environment.md) 已创建并用旧 wheel 完成 CPU 负对照；源码 wheel
成功后可由同一入口在新环境离线安装、核对 native payload 并运行默认/过滤对照。
目前尚未执行成功构建 wheel 的安装分支。

容器预检通过不等于构建成功；wheel 构建成功不等于运行验证或 Hack 验收完成。

## 当前 attempt

容器 `jax-kickoff-cpu-source-002` 已启动，ID 为
`82c9bf9dfe02ba334964a856a5b41c652e83ade2af2ffdb4ce6b950fb4acc077`。
恢复时必须重新 inspect；状态记录只是带时效的快照。
2026-09-14 本轮观察已进入 C++ 编译（LLVM/MLIR 等），尚无成功 wheel/加载验收。
日志：`artifacts/builds/kickoff-cpu-source-002/attempts/0001/build.log`。

旧尝试 `kickoff-cpu-source-001` 实测约使用 2 CPU 后，为利用原有容器额度，发送 SIGTERM
正常停止，再以 4 jobs 创建新尝试。旧 manifest 明确记为 `interrupted`、signal 15；
旧容器已退出，日志完整保留。新尝试复用相同镜像、隔离源码树与持久 Bazel action/cache，
没有扩大容器资源上限。旧尝试不是构建失败或重复运行，恢复时不要重启它。

JAX 隔离树由 shared clone 创建。XLA 的 shared clone 因 partial clone 中无关历史对象缺失
而失败；随后用新 Git 仓库、只读借用原 objects、保留 shallow 边界并 checkout 当前固定
revision，成功得到干净隔离树。没有修改或修复原源码，也没有将缺失历史误判为当前树损坏。

构建驱动使用预检通过的 Python 3.12.3；Bazel 日志还下载了 hermetic Python 3.12.13，
二者职责不同。后续要记录实际外部依赖闭包（包括 hermetic Python、LLVM/MLIR 来源），
不能仅凭驱动环境固定就声称所有构建输入都与五个本地 checkout 一致。

## 核心归档与实际补丁审计

已独立验证 [build-dependency-results.json](build-dependency-results.json)，原始材料在
`artifacts/jax-stack/build-dependencies-002`。XLA 声明的三份基础 revision 都与 baseline
一致；实际编译输入还包括它的补丁集，不能只记录 pristine Git revision。

| 组件 | 固定基础 revision | XLA 补丁数 | 复放并逐字节核对的目标文件数 |
|---|---|---:|---:|
| LLVM | `ab547095ead5464dc024d66264d9b8a987f429f3` | 6 | 24 |
| StableHLO | `7b1b15781ccbd770f50c7eef4b0c3e03834649fd` | 1 | 16 |
| Shardy | `eb23a98329aa70d991aa2d8a51a209af1f8df8fc` | 1 | 3 |

审计完整校验三份下载归档的 SHA-256，只提取补丁涉及的目标文件，在独立 replay
目录按声明顺序应用补丁，再与 Bazel 实际目录逐字节比较。LLVM configure overlay 的
代表源码路径也解析到了经补丁的 llvm-raw。原始 checkout、运行中的源码与配置均未改动。

仓库下载缓存快照中 1,085 个 payload、731,484,721 bytes 的 SHA-256 都与 CAS key 一致。
这是已缓存对象快照，可能包含未使用依赖；未逐个重建所有 external 文件，也没有证明
完整 action 输入闭包或离线可重放。wrapper 仍使用 `--lockfile_mode=off`。

2026-09-14 15:48 UTC 的进程采样记录了 3 个实际 Clang 编译进程，executable SHA-256
均与镜像预检的 Clang 18.1.3 一致；记录与当前构建 CID 绑定。编译 LLVM 的 AMDGPU/SPIRV
源码不代表执行了 GPU 工作负载，这些 target 来自 LLVM configure 的构建集合。

首次审计使用 GNU patch 的 `--fuzz=0`，在固定补丁的不对称尾部 context 上失败；
同一原始 hunk 与文件可精确匹配。后改用完整 supplied context 的 `git apply`，并要求
最终字节与实际 Bazel 源码相等。失败日志保留在 `build-dependencies-001`，没有将其误报为
源码损坏或构建失败。独立验证器不重复执行补丁，而是检查归档、声明、补丁、capture
清单、复放结果与当前编译目录的一致性。

```bash
.venv/bin/python -B research/jax-stack/verify_build_dependencies.py --selftest
```

验证器另用 6 个反例检查损坏哈希、缺失产物、错误证据层级、错误 pin/归档/补丁目标。
这份审计为 `SOURCE-ONLY`，不计为 wheel 成功、安装加载或源码 Hack 验收。
