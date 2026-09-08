# 固定项目环境与跨机器同步

状态：实现与验收完成。用户要求先修复本机环境差异，再把同步过程固定到项目，使用脚本、uv 和 Docker。

## 问题与边界

- 本机缺少 XLA、StableHLO、Shardy、LLVM 四个核心 submodule；JAX 已在固定 commit。
- Python/uv/jaxlib 版本可用，但缺少 Bazel、Clang，源码树残留 ignored Python bytecode。
- 本机 Ubuntu Python 3.12.3 的发行版修订和字节指纹不同于原构建门禁；不替换宿主机系统 Python，也不放宽现有构建哈希检查。
- 保留用户已有文件和 Git index；不执行 Q005、正式 jaxlib 构建或全量可选源码下载。

## 实施方案

1. 补齐并验证五个核心 source root；同步入口从现有 lock/gitlink 读取 revision，拒绝已有改动，且不自动暂存。
2. 建立项目环境锁，固定 Linux x86_64 的 uv、Bazel、容器基础镜像和系统软件包来源及指纹。
3. 提供幂等的宿主机同步/检查入口：初始化源码、验证下载、执行 `uv sync --locked`、清理可再生字节码、运行恢复门禁。
4. 提供 Docker 构建及运行入口，在容器内恢复严格工具链；容器使用独立的 venv 和 cache，源码和产物保持仓库路径一致。
5. 把入口、主机与容器的支持边界、失败恢复方法加入现有 README、构建说明和交接状态。

## 验证与验收

- 核心 submodule HEAD 与 gitlink/lock 一致，根 index 和用户文件保持原状。
- 宿主机 project-status、历史 evidence、baseline verify 和当前 CPU probe 通过，`VERSION-SKEW` 保留。采集新证据时的 live gate 不变。
- 同步脚本的隔离测试覆盖已有改动、错误 revision/hash、路径逃逸、失败返回码和重复执行。
- Dockerfile 实际构建；容器内 uv lock、baseline、严格 preflight 通过。任何外部服务或权限阻塞单独记录，不把配置静态检查写成运行通过。
- 完成后更新本计划、PLAN/HANDOFF/status；按仓库里程碑协议提交并推送本次相关改动。

## 已完成

- [x] 五个核心源码树均恢复到固定 gitlink。
- [x] 将 218 个 ignored bytecode 文件移到 `/tmp` 备份，后续 Python 命令使用 `-B`。
- [x] 完成同步脚本和环境锁；使用直接获取 commit 的方式恢复中断下载。
- [x] 完成 Docker 环境与运行入口，实际构建并验证镜像工具指纹。
- [x] 宿主机与容器 baseline、历史 evidence、当前 CPU Lab 和恢复状态通过。
- [x] 容器严格 preflight 为 `strict-ready: True`；17 项同步隔离测试通过。
- [x] 更新 README、构建说明、HANDOFF 与 PLAN。
- [x] 机器状态校验及 20 个状态负例通过；提交记录以 Git 历史为准。
- [x] 容器重复同步无安装变更，附加命令和独立 venv 挂载验证通过；宿主机 notebook 保留。
