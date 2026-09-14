# 自定义编译 pass 区间：补丁已准备，尚未编译加载

对应 R04/R12。可审查补丁为 [xla-hlo-pass-events.patch](xla-hlo-pass-events.patch)，
结果为 [pass-event-patch-results.json](pass-event-patch-results.json)。当前仅完成
`SOURCE-ONLY` 的补丁生成、应用/反向应用与字节核对，**没有把它应用到正在运行的构建 clone**。
`native_compiled`、`patched_binary_loaded`、`custom_event_observed` 均为 false。
后续运行使用的 [事件验收脚本与负对照](pass-event-acceptance.md) 已完成 CPU 检查，
包括 cold/warm/filter、构建状态拒绝和 wheel/native payload 身份约束。

## 为什么放在 RunHelper 周围

固定 [HloPassPipeline::RunPassesInternal](../../upstream/xla/xla/hlo/pass/hlo_pass_pipeline.cc#L141)
原有的 `TraceMe(pass->name())` 在运行时 filter 判断前创建，作用域延伸到 dump、metadata
和 invariant checks。它能观察整个 pass 外层工作，但过滤掉的调用也可能已有外层事件。

新事件名为 `research_hlo_pass_run`。补丁在 filter 的 `continue` 之后，只对
`!pass->IsPassPipeline()` 的 leaf pass 创建 TraceMe；用局部 lambda 的 RAII 生命周期
包住 `RunHelper` 和结果元数据追加，返回原来的 `StatusOr<bool>`。既有错误处理和 dump
仍在作用域外。它不修改 HLO，不改变 pass 返回值或现有启用/禁用条件。

| 元数据 | 含义 |
|---|---|
| pass / pipeline | leaf pass 与包含它的 pipeline 名称 |
| module / program_id | 开始调用时的模块身份；不假设 pass 后模块 identity 不变 |
| status | StatusCodeToString，例如正常返回时的 OK |
| changed | `true`、`false`；错误返回时为 `unknown` |

[TraceMe 析构调用 Stop](../../upstream/xla/third_party/tsl/tsl/profiler/lib/traceme.h#L180)，
因此正常和 Status 错误返回都结束局部区间。它标记 host 上的编译工作，不是设备运行。
区间还包含追加元数据的开销；正式比较性能时须另测 instrumentation 开销，不能称其为
零开销或纯计算时间。

## 已完成的可逆性检查

capture `artifacts/jax-stack/compiler-event-patch-002` 保留原文件、patched 文件、补丁、
检查命令、日志与清单。原文件 SHA-256 为
`82bb36b51a0cdf76a37d8d4ed7203da7e869d0b991e827b839cf82dae7a1c5df`，
基础 XLA revision 为 `496bd4bd49db9ecbffd85da630b49c860b724604`。

1. 在可丢弃文件副本中通过 `git apply --check`，实际应用后逐字节等于预期 candidate。
2. 通过 `git apply --reverse --check`，实际反向应用后逐字节恢复原文件。
3. 独立 verifier 核对 8 个登记产物、发布补丁与已验证补丁的一致性，以及 scope/状态字段；
   另在新临时文件树实际应用发布补丁，比较 candidate 字节，再反向应用确认恢复。
4. 已有 `compiler-events-001` trace 中，该事件计数为 0。它只是旧 VERSION-SKEW wheel
   的历史负对照，不能替代将来的“同一固定源码、不打补丁”的基线。

首次 capture 001 的普通 unified diff 能应用和回滚，但构建工具要求更严格的字节格式。
capture 002 使用固定 Git objects、可丢弃 index 和独立 work tree 生成规范的
`git diff --binary --full-index --no-color --no-ext-diff --no-textconv`。
原始源码树与原 index 未修改；首次记录保留，没有伪装成可直接通过构建 wrapper 的补丁。

## 后续构建、验证与回滚

先等待 `kickoff-cpu-source-002` 真正结束并完成无补丁 wheel 的安装/加载/数值验收，再在
隔离的 XLA 构建树中应用此补丁。当前容器运行时不要执行以下应用步骤：

```bash
git -C artifacts/jax-stack/source-build-001/clones/xla apply --check \
  "$PWD/research/jax-stack/xla-hlo-pass-events.patch"
git -C artifacts/jax-stack/source-build-001/clones/xla apply \
  "$PWD/research/jax-stack/xla-hlo-pass-events.patch"
```

重建使用新的 build ID、相同固定镜像和缓存，并给现有 `tools/build-jaxlib.py` 增加：

```text
--source-patch=xla=research/jax-stack/xla-hlo-pass-events.patch
```

wrapper 不代为应用补丁；它要求当前 source diff 与给定 patch 字节完全相等，再从固定
revision 复放检查。真正的 wrapper 完整预检与编译尚待执行，不能用本轮副本检查代替。
运行环境应隔离，记录 loaded wheel/native payload 与该 build manifest/patch 的关系。

验收需比较无补丁、补丁、回滚后三个状态：相同输入数值；冷编译中出现带所列 metadata 的
事件；过滤掉的 leaf pass 不出现该事件；warm execution 不产生新的编译 pass 区间。
统计时按 module/program、pipeline/pass、thread 分组，避免父子 inclusive time 重复计数。
设备执行区间仍走另一个入口，不能把这份 C++ host TraceMe 当作 TPU event.start/stop。

回滚通过同一 patch 的 `git apply --reverse --check` 与 `git apply --reverse` 完成，核对
源文件恢复原 hash，并切回已验证的无补丁 wheel。源码回滚、已加载进程的二进制和新进程
实际加载身份必须分别确认。

```bash
.venv/bin/python -B research/jax-stack/prepare_pass_event_patch.py \
  --output artifacts/jax-stack/compiler-event-patch-new
.venv/bin/python -B research/jax-stack/verify_codegen_and_patch.py --selftest
```
