# 自定义编译 pass 区间：源码构建与实际加载已通过

对应 R04/R12。可审查补丁为 [xla-hlo-pass-events.patch](xla-hlo-pass-events.patch)，
历史准备结果为 [pass-event-patch-results.json](pass-event-patch-results.json)，其证据仍为
`SOURCE-ONLY`。新增 [带测试的完整补丁](xla-hlo-pass-events-with-tests.patch) 已应用到
独立 XLA clone，25 个原生 C++ 测试全部通过，见 [首版测试结果](pass-event-cpp-results.json)。
实际运行发现导出器隐藏 `program_id`，因此最终使用
[可导出身份的补丁](xla-hlo-pass-events-exportable.patch)，25 个测试再次通过。
该补丁的 wheel 已实际加载：默认/过滤分别捕获 127/124 个自定义事件，数值与 warm
对照通过。源码已反向恢复；完整三状态验收见 [pass-hack-results.json](pass-hack-results.json)。
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
| module / program_id | 开始调用时的模块身份；program_id 保留在 XSpace |
| research_program_id | 相同 ID 的自定义副本，供 Chrome trace 导出和分组使用 |
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

## C++ 测试及其依赖

完整补丁修改原有 `hlo_pass_pipeline.cc`、`hlo_pass_pipeline_test.cc` 和该目录 `BUILD`；
核心 TraceMe 改动与历史补丁相同。四个新增用例覆盖 changed/no-op 与嵌套 pipeline、
过滤后不发事件、错误返回保留 INTERNAL/unknown 并结束事件、recorder 关闭时 pass 返回值。
测试使用既有 `HloHardwareIndependentTestBase`，实际 XML 为 25 tests、0 failures/errors。

第一次构建失败于既有测试的 `ASSERT_OK_AND_ASSIGN`，没有执行任何测试。JAX 根
`MODULE.bazel` 选用 Googletest `1.17.0.bcr.2`；本次实际依赖不带 XLA 所需宏。
固定 XLA 的 `MODULE.bazel` 和 `third_party/googletest/README.add-status-macros.md`
说明了该宏补丁及 include/circular-dependency 约束。

第二次测试在独立依赖副本应用固定 XLA 的两个 Googletest 补丁，并只向测试命令传入
`--override_module=googletest=...`。250 个原始文件已记录哈希，3 个文件发生变化；
独立反向恢复及脚本复放与实际测试输入逐字节相同。完整日志和 XML 保存到
`artifacts/jax-stack/pass-event-build-001`。它是 JAX 所选版本加测试补丁，不冒充 XLA
独立根模块的完整依赖环境；生产 wheel 命令不包含这个 override。

复现测试依赖副本（输出目录必须尚不存在）：

```bash
python3 -B research/software-stack/prepare_cpp_test_dependency.py \
  --base-directory artifacts/jax-stack/pass-event-build-001/googletest-base-restored \
  --output artifacts/jax-stack/pass-test-dependency-new
```

在固定镜像与原 Bazel 参数下对 `@xla//xla/hlo/pass:hlo_pass_pipeline_test` 执行 `bazel test`，
增加脚本打印的 override 参数；实际完整 argv 见 `test-attempt-002-launch.json`。

最终补丁另检查 `research_program_id`，复跑的 25 个测试仍全部通过。该次日志、XML、
完整 argv 与原文件/candidate 字节保存在 `artifacts/jax-stack/pass-event-build-002`。

## 真实运行发现的导出边界

首版 patched wheel 的原始 XSpace 有 127 个 `research_hlo_pass_run`，每个都包含
`program_id=0`；导出 JSON 保留 127 个事件，却全部省略该字段，验收因此失败。
用本次构建的 protoc 和固定 `xplane.proto` 解码的记录保存在 `program-id-loss.json`
及 `xspace-decode.json`，没有把失败 capture 改成成功。

固定源码的 [IsInternalStat](../../upstream/xla/xla/tsl/profiler/utils/xplane_schema.cc#L615)
把 `kProgramId` 归为内部字段，[导出器](../../upstream/xla/xla/tsl/profiler/convert/xplane_to_trace_events.cc#L94)
明确跳过内部 stat。最终补丁保留原字段，并增加非保留名 `research_program_id`。
真实新 trace 的每个自定义事件都携带该 ID；统计结果中的 `program_id` 分组由这个
可见字段取得，不凭 module 名或事件顺序补造身份。

## 后续构建、验证与回滚

无补丁 003 的安装/加载/数值验收已通过。测试及补丁仅应用到隔离 XLA 构建树。
以下是新实验的应用步骤；现有 clone 已反向恢复，执行前须确认没有构建使用它：

```bash
git -C artifacts/jax-stack/source-build-001/clones/xla apply --check \
  "$PWD/research/software-stack/xla-hlo-pass-events-exportable.patch"
git -C artifacts/jax-stack/source-build-001/clones/xla apply \
  "$PWD/research/software-stack/xla-hlo-pass-events-exportable.patch"
```

重建使用新的 build ID、相同固定镜像和缓存，并给现有 `tools/build-jaxlib.py` 增加：

```text
--source-patch=xla=research/software-stack/xla-hlo-pass-events-exportable.patch
```

wrapper 不代为应用补丁；它要求当前 source diff 与给定 patch 字节完全相等，再从固定
revision 复放检查。最终构建 `kickoff-cpu-pass-events-002` 已完成增量编译和实际加载。
同时保留 001 的成功 wheel 与其运行时字段验收失败记录。
运行环境应隔离，记录 loaded wheel/native payload 与该 build manifest/patch 的关系。

验收需比较无补丁、补丁、回滚后三个状态：相同输入数值；冷编译中出现带所列 metadata 的
事件；过滤掉的 leaf pass 不出现该事件；warm execution 不产生新的编译 pass 区间。
统计时按 module/program、pipeline/pass、thread 分组，避免父子 inclusive time 重复计数。
设备执行区间仍走另一个入口，不能把这份 C++ host TraceMe 当作 TPU event.start/stop。

回滚通过同一 patch 的 `git apply --reverse --check` 与 `git apply --reverse` 完成，核对
源文件恢复原 hash，并切回已验证的无补丁 wheel。源码回滚、已加载进程的二进制和新进程
实际加载身份必须分别确认。

```bash
.venv/bin/python -B research/software-stack/prepare_pass_event_patch.py \
  --output artifacts/jax-stack/compiler-event-patch-new
.venv/bin/python -B research/software-stack/verify_codegen_and_patch.py --selftest
```
