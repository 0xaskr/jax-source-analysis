# 编译 pass 事件验收：源码基线、补丁和回滚

对应 R04/R12。[生产脚本](pass_events_probe.py)、[事件规则](pass_event_checks.py) 和
[验证器](verify_pass_events.py) 已实际运行；[结果清单](pass-events-results.json) 保存
历史 CPU 负对照及一次构建身份拒绝记录。最终补丁的实际构建、加载及回滚证据另保存在
[pass-hack-results.json](pass-hack-results.json)，统一复查入口是 [verify_pass_hack.py](verify_pass_hack.py)。
补丁位置、生命周期和回滚步骤见 [pass-event-patch.md](pass-event-patch.md)。
[配套 Notebook](compiler-pass-hack.ipynb) 的 5 个代码单元已在真实 Jupyter 内核执行，
复查保存的三状态产物、原始 trace、数值参考和导出字段失败；它不重新编译或替换内核的 wheel。

## 匹配源码的三状态对照

使用相同输入，分别在独立环境与进程中加载无补丁 003、最终 patched 002，再加载保留的
无补丁 003。每组都核对 build manifest、Git identity 和已加载 native payload 字节。

| 状态 | 自定义事件（默认/过滤） | 自定义 algsimp（默认/过滤） | generic algsimp（默认/过滤） |
|---|---:|---:|---:|
| 无补丁源码基线 | 0 / 0 | 0 / 0 | 3 / 3 |
| 带补丁源码 wheel | 127 / 124 | 3 / 0 | 3 / 3 |
| 源码恢复、重新加载无补丁 wheel | 0 / 0 | 0 / 0 | 3 / 3 |

各组都只有一次 Python tracing，三次 warm 执行没有新编译事件；输出通过独立 NumPy
参考。补丁组是 `RUN-CPU + SOURCE-PATCHED`，其余为 `RUN-CPU`，均无 `VERSION-SKEW`。
回滚包含三个源文件的原始字节恢复、clone 干净状态，以及新进程实际加载基线 wheel；
没有将已加载进程切换或第三次重建 wheel 当作回滚步骤。

最终 trace 使用 `research_program_id` 保留可导出的模块实例身份；内部 `program_id`
在原始 XSpace 存在，但会被导出器跳过。首版补丁因此真实失败，其 capture 成为验证器
的回归反例。完整来源与修正见 [补丁说明](pass-event-patch.md)。

## 历史旧 wheel 负对照

两组历史 capture 是 `artifacts/jax-stack/pass-events-unpatched-003` 与
`artifacts/jax-stack/pass-events-unpatched-filtered-003`。它们使用相同的生产脚本、
事件规则、已加载 native payload 和输入，分别在独立进程中执行默认编译与禁用 algsimp。
运行环境为现有 jaxlib 0.11.1，保留 `RUN-CPU + VERSION-SKEW`。

| 观察 | 默认 | 禁用 algsimp |
|---|---:|---:|
| generic algsimp 事件 | 3 | 0 |
| generic constant_folding 事件 | 2 | 2 |
| generic layout-assignment 事件 | 1 | 1 |
| research_hlo_pass_run 自定义事件 | 0 | 0 |
| Python tracing / warm 执行次数 | 1 / 3 | 1 / 3 |
| 三次输出最大绝对误差 | 5.56e-9 以下 | 5.56e-9 以下 |

输入为 `A[64,128]`、`W[128,64]`，float32，固定种子 20260915；函数为
`tanh(matmul(A,W) * 1 + 0)`，dot precision 为 HIGHEST。独立 NumPy float64 参考为
`tanh(A @ W)`，容差 `rtol=atol=2e-5`。这些是机制实验，不是实际推理业务验收。

原始 XSpace、导出的 Chrome trace、StableHLO、优化后 HLO、输入输出数组、环境身份、
生产脚本和哈希都保存在 capture 中。两组各有 14 个登记产物。验证器重新解析原 trace、
计算数值误差、比较二进制身份和输入，并重新推导事件观察。

旧 wheel 禁用 algsimp 后 generic 事件为 0，只是本次运行观察。固定源码的 generic
TraceMe 在 filter 之前创建（见补丁说明），因此未来匹配源码的过滤对照**不要求 generic
计数为 0**；要求消失的是放在 filter 之后的自定义 leaf pass 事件。

`pass-events-running-build-rejected-002` 在读取到 `kickoff-cpu-source-002` 的状态仍为
`running` 时被入口拒绝，错误为 `build manifest is not a successful real build`。
这份 capture 保存当时读取的完整 manifest 和 SHA-256，后续用快照复查，不依赖活跃
manifest 一直保持 running。它没有进入 JAX 实验或生成 trace，仅为 `SOURCE-ONLY`。

## 哪些条件才允许验收补丁

1. `--expected-events present` 必须指定成功的真实 build manifest；dry run、运行中或
   失败构建均拒绝。构建记录须包含唯一的 XLA patch，其 SHA-256 等于本研究发布的补丁。
2. 复用仓库构建验证器核对历史 manifest、构建产物及 wheel；JAX/XLA revision 必须与
   baseline pin 一致。Python JAX 必须来自固定 editable checkout。
3. 逐一核对环境捕获的已加载 jaxlib `.so`：当前文件的大小和 SHA-256 必须与记录一致，
   也必须与构建 wheel 中对应成员相等，至少包括 `_jax.so`、`libjax_common.so`。
   相同 version 或 Git revision 不能替代 native payload 核对。
4. raw trace 必须包含 lower、一次 Python tracing、cold compile 和三次 warm 范围。
   cold 必须捕获已知 native pass，避免把未启用 profiler 当作“没有事件”。
5. 自定义事件须携带 pass、pipeline、module、research_program_id、status、changed；目标模块
   事件须位于 cold 范围，可在 compiler worker 线程上，但须由同线程 generic pass 包围。
   默认编译须有自定义 algsimp；过滤后不得有它；编译事件不得与 warm 执行重叠。
6. 默认/过滤两组须使用相同输入、native payload、构建来源、生产脚本和规则，输出都
   通过独立数值参考。两组都满足正对照条件，才设置 `patched_runtime_pair_accepted`。

事件规则按 module/program_id、pipeline/pass、thread 分组，分别报告 inclusive duration
之和与区间并集。Chrome trace 的单位是微秒；这些时间含 instrumentation 开销，不是
设备时间，也不能把跨线程或嵌套 duration 相加当作编译总耗时。

最终 patched pair 的 `source_build_verified` 和 `patched_runtime_pair_accepted` 均为 true。
历史旧 wheel pair 保持 false。脚本使用研究 capture 记录验收，不修改
构建 wrapper 中目前固定为 `not-run` 的 `runtime_validation` 字段。

## 复现与后续运行

在仓库根目录执行，输出目录必须尚不存在，且应先清除 ambient `XLA_FLAGS`：

```bash
.venv/bin/python -B research/software-stack/pass_events_probe.py \
  --output artifacts/jax-stack/pass-events-negative-new --expected-events absent
.venv/bin/python -B research/software-stack/pass_events_probe.py \
  --output artifacts/jax-stack/pass-events-negative-filtered-new \
  --expected-events absent --disable-pass algsimp
.venv/bin/python -B research/software-stack/verify_pass_events.py \
  --default artifacts/jax-stack/pass-events-negative-new \
  --filtered artifacts/jax-stack/pass-events-negative-filtered-new --selftest
```

无补丁源码 wheel 成功并装入独立环境后，使用该环境的 Python 运行同样的 absent 两组，
增加 `--jaxlib-build-manifest manifests/build-fingerprints/kickoff-cpu-source-003.json`。
这一步才是匹配源码 baseline。随后按补丁说明构建新 wheel，在另一个独立环境中运行
present 两组，指向新构建的 manifest。最后反向应用补丁、确认源码恢复，并在新进程
重新加载已验证的无补丁 wheel，重做 absent 两组作为回滚检查。

本次三状态统一复查须在固定构建镜像中执行，以满足 canonical manifest 的 Clang 检查：

```bash
.venv/bin/python -B research/software-stack/verify_pass_hack.py --selftest
```

默认复查已发布证据并运行 selftest：

```bash
.venv/bin/python -B research/software-stack/verify_pass_events.py --selftest
```

`--write` 将指定 pair 的结果写到 `pass-events-results.json`，并附加上述历史 running
拒绝记录。Selftest 固定使用历史旧 wheel 负对照，另验证 8 种坏事件、3 种坏 manifest、
缺失构建身份、worker 线程包含关系与嵌套区间统计。合成事件只测试规则，**不计为补丁
执行证据**。生产脚本和验证器共享事件规则，数值参考和原始文件哈希另行复查。

[Pallas/profiling Notebook](pallas-profiling.ipynb) 包含不可变 capture 复查和新进程重跑。
7 个代码单元已在真实 Jupyter 内核执行通过，包括新进程产生的默认/过滤 capture。
它使用当前环境做负对照；TPU 设备事件的 LLO、设备 trace 与开销仍需单独验收。
