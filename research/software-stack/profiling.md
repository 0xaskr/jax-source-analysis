# XProf：host、编译 pass、设备执行

用户已确认三类区间全部研究（U04）。Host 与现有 CPU wheel 的编译期事件已运行验证；
设备侧已验证开放源码 lowering 产生的 Mosaic 标记。自建编译器补丁已完成 CPU 构建、加载
及三状态对照，见 [pass 事件验收](pass-event-acceptance.md)；TPU 真机事件仍未验收。
结果见 [extension-results.json](extension-results.json)，独立复查入口为
[verify_extensions.py](verify_extensions.py)。该历史 capture 保留 `VERSION-SKEW`；新增
源码基线与补丁事件另有 build/native 绑定，不带该限定。

| 区间 | 插入入口 | 当前证据 | 尚未证明 |
|---|---|---|---|
| Host 函数/请求 | `TraceAnnotation`、`StepTraceAnnotation` | 15 个历史范围；新增 4 个 source-bound 跨线程任务，逆序完成和异常终点通过 | 任意多 host/进程关联、TPU 设备时长 |
| 编译 pipeline/pass | C++ `tsl::profiler::TraceMe`，leaf RunHelper 范围 | 自建补丁默认/过滤 127/124 个事件，algsimp 3→0，数值与 warm 对照通过 | TPU 编译路径覆盖与 instrumentation 开销 |
| Pallas TPU kernel 内部 | kernel 内 `jax.named_scope` → `tpu.trace_start/stop` | 无标记/三组标记对照，真实生产 lowering | libtpu 接受、LLO 保留、TPU 执行与 trace 可见性 |
| 普通 JAX TPU 运算 | 源码名称信息 + 设备 profiler | 固定源码/API 定位 | 不能据名称 metadata 宣称存在任意设备区间的 start/stop |

## Host 起止与异步边界

[TraceAnnotation](../../upstream/jax/jax/_src/profiler.py) 继承 native TraceMe。
实际 nanobind wrapper 位于 [profiler.cc](../../upstream/xla/xla/python/profiler.cc)：
构造时启动，`__enter__` 返回自身，`__exit__` 调用 `Stop()`。
实验特意把工作放在构造与 `__enter__` 之间，导出的父区间确实包含这段工作。
异常退出也关闭范围；profile session 外的标记没有出现在 trace。

通常直接用 context manager。若要在两个函数中分别起止，可用 `ExitStack` 持有 context，
在 `finally` 中 `close()`，完整示例见 [profile_events_probe.py](profile_events_probe.py)。
该历史实验只验证同线程生命周期。新增 [XSpace 关联实验](xspace-contexts.md)
以两个独立本地范围和内部 context 字段连接跨线程起止，包含逆序完成和异常终点；
未把活的 TraceAnnotation 传给另一线程，也未定义稳定的公开异步事件 API。

每步拆成 dispatch 和 `block_until_ready()` 等待。Host 外层测量主机所见的过程；
结束 dispatch 不保证设备完成，profile session 也需要覆盖设备完成点。
官方 [JAX profiling 文档](https://docs.jax.dev/en/latest/profiling.html) 同样要求在示例中等待结果。
在线页面仅作接口说明，版本归属以固定源码与 capture 为准。

`Trace Event JSON` 的 `ts`/`dur` 单位是微秒：
[AddTraceEvent](../../upstream/xla/xla/tsl/profiler/convert/trace_events_to_json.cc) 把 ps 转为 μs。
`displayTimeUnit=ns` 不改变这些数值的单位。统计按事件名称计数、汇总 inclusive duration；
需要经过时间时取区间并集。每步“剩余范围”是父范围减去**选定**子区间并集，
不能称作完整 profiler 的 self time。本实验的耗时含观测开销，不是性能基准。

[普通 TPU 公开运行接口](tpu-runtime-boundary.md) 进一步区分了输出 buffer ready、执行
status 与当前线程 effect tokens；PJRT 库调用使用 type 14 的 linkage context，并不是设备
kernel start/stop。该部分只有源码证据，不增加上述运行计数。

## 编译 pass：冷编译与重复执行分别捕获

[compiler_events_probe.py](compiler_events_probe.py) 关闭持久编译缓存，分别标记
`lower()`、`compile()` 与三次重复执行。Python 函数体里另放一个 `TraceAnnotation`。

- Python tracing 计数为 1，函数体里的 host 标记也只有 1 个，位于 `research_cold_lower` 内。
- 三次 warm scope 内没有本次检查的 `algsimp`、`constant_folding`、`layout-assignment`。
- 冷编译同一线程范围内有 171 个事件，包括 3 次 `algsimp`、2 次 `constant_folding`、
  1 次 `layout-assignment`。这不是“171 个不同 pass”，也没有计入工作线程事件。
- 三份输出均通过 NumPy float64 参考对照，`rtol=atol=2e-5`。

固定 XLA 的 [HloPassPipeline::RunPassesInternal](../../upstream/xla/xla/hlo/pass/hlo_pass_pipeline.cc)
在循环中创建 `TraceMe(pass->name())`。范围还涉及元数据、检查与 dump，且标记在过滤 gate
前创建。因此事件存在不等于转换已执行或 IR 有变化，要联查 before/after IR 与 changed 记录。

诊断补丁已经在 leaf RunHelper 增加 RAII `TraceMe` 与身份 metadata；
固定源码构建、加载、数值对照和回滚均已完成，见 [编译器 Hack](pass-event-acceptance.md)。
上面的历史 wheel 捕获保留原证据边界。Python `jit` 函数体里的标记随 tracing 执行，
不代表 executable 每次执行的内部区间。

## 设备内部：Pallas 名称栈变成 IR 操作

[mosaic_events_probe.py](mosaic_events_probe.py) 使用 `A[16,128] @ W[128,128]`，
输出 tile `[8,128]`，grid `[2,1]`。同一 kernel 分别关闭/开启名称范围：

```python
with jax.named_scope("research_kernel"):
    with jax.named_scope("research_dot"):
        product = jnp.matmul(a_ref[...], w_ref[...], precision="highest")
    with jax.named_scope("research_store"):
        out_ref[...] = product
```

脚本使用 `jax.jit(...).trace(...).lower(lowering_platforms=("tpu",))`。
进程内 observer 调用原生产 lowering、记录原 module 并原样返回，退出时恢复函数；
没有修改源码树。保存 kernel Jaxpr、canonicalize 前的 Mosaic MLIR、外层 StableHLO 和导出 HLO。
没有调用 TPU backend 的 `.compile()`：证据记为 `RUN-CPU`、阶段明确为开放源码 TPU lowering，
不计为 `COMPILE-TPU` 或 `RUN-TPU` 验收。

无 scope 版本有 0 个标记；有 scope 版本产生 3 个 `trace_start` 与 3 个 `trace_stop`，
名称依次是 kernel/dot/store，level 都为 10。独立验证器解析 MLIR 操作、检查块内配对，
没有把 debug location 中重复出现的名称算成事件。两版外层均有 `tpu_custom_call`。
同样形状的 generic interpret 数值通过，只作为 CPU 语义对照。

生成点是 [jaxpr_subcomp](../../upstream/jax/jax/_src/pallas/mosaic/lowering.py) 的 name stack
边界处理；操作定义见 [tpu_ops.td](../../upstream/xla/xla/mosaic/dialect/tpu/tpu_ops.td)。
Mosaic TPU MLIR 不叫 LLO。标记级别、libtpu 编译策略与设备采集配置如何影响最终事件，
当前没有真机材料来确定。

TPU 验收需明确型号和 libtpu 身份后，保存无 scope/有 scope 两版编译产物和设备 trace，
确认事件属于 TPU plane/core，检查名称、起止、嵌套、调用次数、数值与标记开销。
普通 JAX TPU 的 HLO 名称归因也要单列对照。这部分等待 U03，不用 CPU profiler 结果替代。

## 复查与重放

```bash
.venv/bin/python -B research/software-stack/verify_extensions.py --selftest
.venv/bin/python -B research/software-stack/profile_events_probe.py --output artifacts/jax-stack/profile-events-replay
.venv/bin/python -B research/software-stack/compiler_events_probe.py --output artifacts/jax-stack/compiler-events-replay
.venv/bin/python -B research/software-stack/mosaic_events_probe.py --output artifacts/jax-stack/mosaic-events-replay
```

验证器复查已登记的四组 capture，共 62 个产物的大小、SHA-256、完整清单和关键语义；
三个负向用例拒绝伪造哈希、删除清单项、移除 `VERSION-SKEW`。
重放须用新目录；新 capture 需明确登记后才能替换已确认结果。
