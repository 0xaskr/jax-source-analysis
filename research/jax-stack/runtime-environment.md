# 独立运行环境与源码 wheel 加载入口

对应 R04/R12。[准备脚本](prepare_pass_runtime.py) 已实际创建并运行独立 CPU 环境；
[结果](runtime-environment-results.json) 与 [验证器](verify_runtime_environment.py) 保留复制、
导入路径、事件和数值检查。当前完成的是继承旧 wheel 的环境验证，**尚未安装源码构建
wheel，也没有补丁事件执行证据**。

## 已完成的独立环境

环境位于 `artifacts/jax-stack/runtime-environment-001/venv`，入口为其中的 `bin/python`。
脚本先用 `venv --without-pip --copies` 新建环境，再复制宿主机 `.venv` 的 site-packages，
排除 `__pycache__`。没有复制宿主机 bin 下可能带旧路径 shebang 的工具脚本。

实际核对了 12,603 个依赖文件，共 784,588,999 bytes：新文件和宿主机文件大小、SHA-256
相同，inode 不同；复制前后宿主机包清单和内容一致。Python executable 也有独立 inode，
与宿主机实际 Python 的字节相同。这些检查证明当前文件副本的隔离关系。

新 Python 的 `sys.prefix` 指向新 venv，`sys.path` 不含宿主机 `.venv`，user site 被禁用。
实际导入的 jaxlib、NumPy 和捕获的 jaxlib native 文件都位于新环境；JAX 保留固定源码的
editable 安装，实际从 `upstream/jax` 导入。OS 库、Python 标准库与 JAX 源码仍共享，
因此这不是完整隔离的系统镜像或离线依赖闭包。

默认/禁用 algsimp 两组实验在新环境的独立进程中运行，均捕获一次 Python tracing、
一次 cold compile 和三次 warm 执行。generic pass 计数与旧宿主机对照一致，自定义事件
均为 0，三次输出的最大绝对误差低于 5.56e-9。详细 workload 与事件规则见
[pass-event-acceptance.md](pass-event-acceptance.md)。此处仍为 `RUN-CPU + VERSION-SKEW`。

原始目录中包括复制前后清单、解释器信息、命令/日志、两组完整 capture 和运行结果。
源码及运行产物继续保留在 Git 外；后续实验使用新目录，不覆盖这份历史环境，否则已有
capture 的 native 文件复查会失败。

## 一条命令接入成功构建

当前旧 wheel 环境验证的复现命令（输出目录须尚不存在）：

```bash
.venv/bin/python -B research/jax-stack/prepare_pass_runtime.py \
  --output artifacts/jax-stack/runtime-environment-new
```

`kickoff-cpu-source-002` 成功结束后，用以下命令新建环境、离线安装它的 wheel，并运行
匹配源码的默认/过滤 absent 两组。此命令本轮尚未通过源码 wheel 安装分支：

```bash
.venv/bin/python -B research/jax-stack/prepare_pass_runtime.py \
  --output artifacts/jax-stack/runtime-source-baseline-001 \
  --jaxlib-build-manifest manifests/build-fingerprints/kickoff-cpu-source-002.json \
  --expected-events absent
```

安装前复用构建 manifest 的完整验证器，核对成功状态及产物。安装使用核对过 SHA-256
的固定 uv，显式指定新 Python、`--offline --no-index --no-deps --no-build --no-cache
--link-mode=copy --reinstall`。它只替换新环境内的 jaxlib，随后由 pass 实验核对已加载
`.so` 与该 wheel 的成员字节；不能仅凭 version/Git revision 认定安装身份。

补丁构建则使用新的 build manifest、新输出目录以及 `--expected-events present`。
脚本不应用源码补丁，完整生命周期仍按 [pass-event-patch.md](pass-event-patch.md) 执行。
无补丁、补丁和回滚运行使用各自可复查的目录与新进程，保留所有历史 wheel 和 capture。

## 拒绝条件与复查

入口集成测试实际验证了两个失败：`present` 未指定构建 manifest，以及指定的 manifest
状态仍为 `running`。两者均在创建 venv 前拒绝；失败日志和 running 输入快照保存在
`runtime-environment-missing-build-rejected-001`、
`runtime-environment-running-build-rejected-001`，只计为 `SOURCE-ONLY`。

复查保留的旧 wheel 环境：

```bash
.venv/bin/python -B research/jax-stack/verify_runtime_environment.py --selftest
```

该验证器专门检查初始 inherited 环境：元数据哈希、完整依赖清单、当前文件/inode、
解释器身份、事件/数值和两个入口拒绝。它会拒绝三种错误元数据清单，且不修改原始证据。
未来源码 wheel 的默认/过滤 pair 使用 `verify_pass_events.py --default ... --filtered ...`
复查，不把初始环境验证当作安装分支已经执行。

## 完整 CPU 实验的源码身份与复验

matmul、fusion/memory 和 overlap 生产脚本现均支持 `--jaxlib-build-manifest`，通过
[capture_runtime.py](capture_runtime.py) 复用成功构建 gate 与 native/wheel 字节核对。
指定构建时只接受无补丁源码基线；补丁事件实验继续使用专门的 pass 入口。命令记录
改为实际 `sys.executable` 和进程启动参数，能够区分宿主机与独立环境。

不指定构建记录时，baseline 采集器也能识别仓库内的本地 wheel；但本地文件路径本身
不等于完整构建来源或已加载 native payload 绑定。因此源码复验显式传入 manifest，
并保存 `build-binding.json`；相应验证器会重新核对，缺少或不一致的选择会被拒绝。

已在保留的独立旧 wheel 环境实际执行并复查 [CPU 复验结果](cpu-revalidation-results.json)：

| Capture | 数值对照 | 产物 |
|---|---:|---:|
| cpu-matmul-runtime-001 | 4 组，含 executable 同进程序列化重载 | 1,063 |
| fusion-memory-runtime-001 | 11 组 | 2,029 |
| overlap-runtime-001 | 3 组，以及 4 个重新观察的失败 | 595 |

共 18 组数值对照、3,687 个产物，仍为 `RUN-CPU + VERSION-SKEW`。三个 CLI 的 running
构建拒绝已实际测试，发生在运行采集前；另用两个反例检查构建选择缺失与不一致。
记录保存在 `artifacts/jax-stack/source-revalidation-gates-001`。这不是成功构建分支的证据。

无补丁源码环境成功建立后，以下命令执行完整 matmul，并复查实际加载身份及原始产物：

```bash
artifacts/jax-stack/runtime-source-baseline-001/venv/bin/python -B \
  research/jax-stack/matmul_probe.py \
  --output artifacts/jax-stack/cpu-matmul-source-001 \
  --jaxlib-build-manifest manifests/build-fingerprints/kickoff-cpu-source-002.json
artifacts/jax-stack/runtime-source-baseline-001/venv/bin/python -B \
  research/jax-stack/verify_research.py --capture artifacts/jax-stack/cpu-matmul-source-001
```

fusion/memory 和 overlap 同样传入该 manifest，在各自新目录采集；对应验证器均支持
`--capture`。overlap 源码复验另外使用 `--observe-prior-failures`，记录四个历史失败
用例的本次 lowering、编译或执行结果。若某个用例现在成功，必须执行并通过独立 NumPy
参考；数值错误仍会终止采集。当前旧 wheel 中四个用例仍失败，历史用例转为成功的分支
尚未实际触发，不能宣称修复。

fusion 验证器对已绑定源码构建的 capture 重新报告 reduction peak 是否仍与 inclusive
logical max 不同，保留原始 allocation、live-range 和数值约束。它不要求新 binary
继续出现旧诊断差异，也不会据此推导物理峰值或 TPU 内存变化。旧 capture 的严格历史
检查仍可复现。

本轮环境检查另发现上游源码树中有 305 个忽略的 `.pyc/.pyo` 文件。调用现有
`tools/sync-environment.py` 的 `quarantine_bytecode` 将它们移动到
`artifacts/environment/bytecode/backup-ai8r8u6z`，全部备份字节核对通过；记录在
`artifacts/jax-stack/bytecode-quarantine-001/record.json`。没有修改源码文件、运行中的
构建 clone 或环境锁。之后 `sync-environment.py check` 与 17 项环境工具自测通过。
执行 Jupyter 时须让内核进程继承 `PYTHONDONTWRITEBYTECODE=1`；仅在启动 nbclient 的
父进程加 `-B` 不能保证新内核也禁写字节码。
