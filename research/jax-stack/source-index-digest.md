# 源码与 API 索引导读（由 source-index.json 生成）

本文件由 `render_source_index.py` 从 [source-index.json](source-index.json) 生成，
不是手工维护的副本；两者不一致时应重新生成而不是手改。

- kickoff revision: 51
- schema_version: 1.0
- 入口总数: **219**，关系总数: **92**

## 组件覆盖

| 组件 | 入口数 | 关系（出/入） | 职责 |
|---|---|---|---|
| `jax` | 54 | 23/17 | Python 前端：API、tracing、Jaxpr、primitive 规则与 MLIR lowering |
| `stablehlo` | 1 | 0/0 | 可移植方言定义：操作、类型与属性 |
| `shardy` | 8 | 4/5 | 分片方言与 import/propagation/export pass 组织 |
| `xla` | 141 | 60/64 | MLIR/HLO 转换、后端优化、调度、buffer assignment 与代码生成 |
| `llvm` | 4 | 1/3 | CPU 代码生成：LLVM IR、MC、ORC JIT |
| `xprof` | 11 | 4/3 | 性能数据采集、转换与聚合 tooling |

各固定 revision：

| 组件 | revision |
|---|---|
| `jax` | `5832e866449a41c3eea6333416528039119a0fde` |
| `llvm` | `ab547095ead5464dc024d66264d9b8a987f429f3` |
| `shardy` | `eb23a98329aa70d991aa2d8a51a209af1f8df8fc` |
| `stablehlo` | `7b1b15781ccbd770f50c7eef4b0c3e03834649fd` |
| `xla` | `496bd4bd49db9ecbffd85da630b49c860b724604` |
| `xprof` | `68dba1826c37986af41f6119930ec315592a51f6` |

## 证据级别分布

| 证据级别 | 入口数 |
|---|---|
| `SOURCE-ONLY` | 219 |

> `SOURCE-ONLY` 表示只核对了固定源码，没有运行产物。索引条目的证据级别描述的是
> **该入口被核对的方式**，不是整个研究目标的完成状态。

## 索引声明的边界

- Initial entry-point index; the complete caller/callee graph is still in progress.
- libtpu private implementation, LLO and target hardware internals are not indexed.
- Selected LLVM optimization/MC/ORC interfaces and CPU kernel invocation are indexed; the complete internal pass graph and hardware implementation remain outside current coverage.

## 入口清单（按组件与层次）

### `jax` — 54 个入口

| 层次 | 符号 | 位置 | 输入 → 输出 | 约束 | 关系 |
|---|---|---|---|---|---|
| CPU executable and runtime | `persistent_id` | `upstream/jax/jax/experimental/serialize_executable.py#L93` | Python 包装中的 executable 对象 → exec persistent id 与序列化 bytes | 这里只读取该字节模式；没有调用 Unpickler 或 native loader。 | — |
| CPU interpreter | `pallas_call_hlo_interpret` | `upstream/jax/jax/_src/pallas/hlo_interpreter.py#L316` | Pallas kernel Jaxpr、grid 和输入数组 → 状态消解后的 HLO 解释循环结果 | interpret=True 是通用解释路径；不能等同于 TPU 专用语义模拟。 | — |
| Collective lowering | `_emit_async_start_custom_call` | `upstream/jax/jax/_src/lax/parallel.py#L3055` | target、future aval、collective config → 携带 async_collective_config 的 custom call | Future lowering 使用 inner_aval 的 tensor 类型；不同于 FFI 默认把 Future shape 当作空。 | — |
| Collective tracing | `_psum` | `upstream/jax/jax/_src/lax/parallel.py#L155` | array、axis、axis groups、is_async → psum primitive 或 invariant async primitive | check_vma=True 分支传播 is_async；False 分支直接绑定同步 psum_p。 | — |
| Cost binding | `BuildXlaCompilerSubmodule` | `upstream/jax/jaxlib/xla_compiler.cc#L86` | hlo_module_cost_analysis 的 client 与 HloModule → properties dict | 从该 client 的 PJRT backend 获取 analyzer 并遍历 entry computation；没有 Python rates 参数。 | ↓`xla.cpu-cost-factory` |
| Donation lowering | `_set_up_aliases` | `upstream/jax/jax/_src/interpreters/mlir.py#L1487` | donated arguments、输入输出 avals、memory kinds、layouts/shardings → input-output aliases 与交给 XLA 的 donation 标记 | 优先精确匹配，另有相同元素数量的 fallback；最终 backend 合法性、临时量和运行时复用仍需检查。 | ↑`jax.mlir-module` |
| Executable packaging | `serialize` | `upstream/jax/jax/experimental/serialize_executable.py#L27` | Compiled 对象 → 含 PJRT executable 的 pickle 包与 pytree 信息 | 离线审计只读原始字节，在包中定位完整 .o；没有 unpickle 或重新加载历史文件。 | — |
| Experimental scheduling | `control_dep` | `upstream/jax/jax/experimental/overlap.py#L18` | src 与 dst → 有副作用、无输出的 FFI control_dep | 当前 Future 在 VMA 或默认 layout 上存在已捕获失败；不能直接视为可用 async overlap API。 | ↑`jax.overlap-schedule` |
| Experimental scheduling | `schedule` | `upstream/jax/jax/experimental/overlap.py#L23` | 按顺序排列的值 → 相邻值的 control_dep custom calls | 只是控制提示；VMA、layout、backend 重写和合法性仍须逐阶段验证。 | ↓`jax.overlap-control` |
| FFI layout | `_aval_shape` | `upstream/jax/jax/_src/ffi.py#L179` | FFI operand aval → 用于默认 layout 的 shape | AbstractFuture 返回空 shape；本次 rank-2 future lowering 实际为 tensor，触发 MLIR layout 验证错误。 | — |
| Jaxpr to MLIR | `lower_jaxpr_to_module` | `upstream/jax/jax/_src/interpreters/mlir.py#L1324` | Jaxpr 与平台/设备、effects、alias/sharding 等 lowering 上下文 → LoweringResult，包含 MLIR module | primitive lowering 使用平台注册规则；生成表示不同于执行后端编译。 | ↑`jax.cached-mlir-lowering` ↓`jax.donation-aliases` |
| Lowering | `_cached_lowering_to_hlo` | `upstream/jax/jax/_src/interpreters/pxla.py#L717` | Jaxpr、输入输出 avals、设备和别名信息 → lower_jaxpr_to_module 的 lowering 结果 | 函数名含 hlo，但这里创建 MLIR module；不要据名称把它当成 XLA HloModule pass pipeline。 | ↓`jax.mlir-module` |
| Lowering | `lower_sharding_computation` | `upstream/jax/jax/_src/interpreters/pxla.py#L976` | Jaxpr、mesh、sharding/layout 与平台 → 可继续 compile 的计算对象 | 需要区分多设备/效果约束；单 CPU 示例不能证明多设备 sharding 策略。 | ↑`jax.pjit-lower` |
| Lowering | `_pjit_lower` | `upstream/jax/jax/_src/pjit.py#L1224` | Jaxpr、sharding/layout、donation、平台与编译参数 → pxla.MeshComputation | 把这些参数交给 lower_sharding_computation，不直接输出机器码。 | ↓`jax.lower-sharding` |
| Metadata API | `set_xla_metadata` | `upstream/jax/jax/_src/xla_metadata.py#L109` | 可选数组值或 kwargs metadata → 标记 producer op 的 identity primitive 或 metadata context | Python 值转成字符串（布尔小写）；属性能被传递不代表 backend 有消费逻辑。 | — |
| Metadata API | `xla_metadata_call2` | `upstream/jax/jax/_src/xla_metadata.py#L215` | 函数、metadata、ad_metadata 策略 → 带 metadata 的 staged call | ad_metadata 控制线性化/转置派生调用；same/drop 的 matmul 反向调用已分别捕获。 | — |
| Metadata lowering | `_xla_metadata_call_lowering` | `upstream/jax/jax/_src/xla_metadata.py#L316` | Jaxpr、operands、metadata → 带 mhlo.frontend_attributes 的 func.call | 属性最初在 call 上；后端内联和 fusion 可能改变属性所有者，不能按节点数量一一对应。 | — |
| Mosaic TPU | `lower_jaxpr_to_pipelined_module` | `upstream/jax/jax/_src/pallas/mosaic/lowering.py#L1051` | kernel Jaxpr、grid、mesh、dimension semantics → Mosaic TPU MLIR Module | 被调用实现检查 libtpu 版本；该 MLIR 不是 LLO。 | ↑`pallas.tpu-lower` |
| Mosaic TPU | `pallas_call_tpu_lowering_rule` | `upstream/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L393` | kernel Jaxpr、GridMapping、CompilerParams、alias/effects → 外层 TPU custom call | 先创建内层 Mosaic module，再通过 _lower_to_custom_call 嵌入外层 module。 | ↑`pallas.dispatch` ↓`mosaic.module` |
| Mosaic TPU lowering | `jaxpr_subcomp` | `upstream/jax/jax/_src/pallas/mosaic/lowering.py#L1746` | kernel Jaxpr、name stack、block shapes 与 MLIR values → 含 tpu.trace_start/stop 的 Mosaic MLIR | name stack 边界产生 level=10 的标记；此开放源码阶段不能证明 libtpu/LLO/设备 trace 保留标记。 | — |
| Mosaic payload boundary | `_tpu_custom_call_lowering` | `upstream/jax/jax/_src/tpu_custom_call.py#L409` | 序列化 kernel config、输入节点、alias/layout 与结果信息 → target=tpu_custom_call 的外层 custom call | backend_config 携带编译载荷；has_side_effect、alias、layout 是接口合同。 | — |
| Pallas API | `pallas_call` | `upstream/jax/jax/_src/pallas/pallas_call.py#L1135` | kernel、out_shape、grid、BlockSpec 与 compiler_params → 可调用的 kernel wrapper | 表达对 Ref/块的操作；CPU 仅有 interpret 路径，不能作为 TPU 编译或硬件验证。 | — |
| Pallas lowering | `_pallas_call_lowering` | `upstream/jax/jax/_src/pallas/pallas_call.py#L844` | kernel Jaxpr、grid 参数和 interpret 选项 → 平台对应的 lowering 结果 | interpret 分支 lower 解释实现；TPU 分支进入 pallas_call_tpu_lowering_rule。 | ↓`pallas.tpu-lower` |
| Primitive API | `dot` | `upstream/jax/jax/_src/lax/lax.py#L2544` | 数组与 dimension_numbers、precision、结果类型/分片 → dot_general primitive 的结果 | precision 和 preferred_element_type 是不同参数；不把数学 dot 等同于某一种硬件指令。 | ↑`jax.dot-general` |
| Primitive API | `dot_general` | `upstream/jax/jax/_src/lax/lax.py#L2528` | lhs/rhs 与收缩/batch 维度 → dot 运算的结果数组 | 此 revision 的 dot_general 是 dot 的包装入口；不能按旧版本假设其内部直接 bind。 | ↑`jax.matmul` ↓`jax.dot` |
| Primitive lowering | `_dot_general_lower` | `upstream/jax/jax/_src/lax/lax.py#L6264` | lowering context、MLIR lhs/rhs、dimension_numbers 和 precision → stablehlo.dot_general 及必要转换 | 构造 DotDimensionNumbers 和 precision_config；输出仍是 MLIR value。 | — |
| Private async collective API | `psum_start` | `upstream/jax/jax/_src/lax/parallel.py#L3120` | 局部 array 与 collective axis → AbstractFuture；具体分支可能返回普通 array | 不是 jax.lax 公开导出；关闭 check_vma 的 _psum 分支没有使用 is_async。 | — |
| Profiling | `TraceAnnotation` | `upstream/jax/jax/_src/profiler.py#L347` | 事件名称及 metadata/context 范围 → host TraceMe 事件 | 构造时启动，__exit__ 停止；Python jit 函数体中的标记随 tracing 执行，不能代表每次设备运行。 | — |
| Public PJRT execution and completion | `block_until_ready` | `upstream/jax/jax/_src/api.py#L2751` | pytree 的 leaves → 同结构、所选 JAX array leaves ready | 单 Array 或批量分支；不把未传入的工作自动纳入等待范围。 | ↓`jaxlib.array-wait` |
| Public PJRT execution and completion | `effects_barrier` | `upstream/jax/jax/_src/api.py#L2747` | 当前线程的 runtime_tokens → 等待已记录 effects | runtime_tokens 为 threading.local；不是跨线程/跨 host 的通用 barrier。 | ↓`jax.runtime-tokens` |
| Public PJRT execution and completion | `__call__` | `upstream/jax/jax/_src/interpreters/pxla.py#L400` | Python 参数、输入/输出 handlers 与 effects → 结果 PyArray 与 token 注册 | ordered/unordered effects 或 host callbacks 时 with_tokens=True；否则默认不向 Python 返回完成 status。 | ↓`jaxlib.execute-sharded` |
| Public PJRT execution and completion | `_get_fastpath_data` | `upstream/jax/jax/_src/pjit.py#L190` | executable、effects、输出与 PGLE → fastpath metadata 或 None | ordered/unordered/ref effects、部分 PGLE/no_execution 和非 Array 输出等拒绝 fastpath；不能把所有 warm 调用统一画到 C++ 快路径。 | — |
| Public PJRT execution and completion | `PjitFunction::Call` | `upstream/jax/jaxlib/pjit.cc#L632` | Python 参数与缓存 executable → 包装 IFRT 输出的 PyArray | 满足 fastpath 条件后直接调用 IFRT Execute，未经过 PyLoadedExecutable::ExecuteSharded；返回 wrapper 不等于设备完成。 | ↓`ifrt.pjrt-execute` |
| Public PJRT execution and completion | `RuntimeTokenSet` | `upstream/jax/jax/_src/dispatch.py#L115` | 线程局部 effect/runtime token → 按 effect/device 保存 token | block_until_ready 等待后清空当前线程记录；不能保证其他线程的 token 被看到。 | ↑`jax.effects-barrier` |
| Public PJRT execution and completion | `PyArray::BlockUntilReady` | `upstream/jax/jaxlib/py_array.cc#L858` | 一个 PyArray → ready 或错误 status | 释放 GIL，调用 AwaitBuffersReady；本入口没有等同于整个运行的 result_status。 | ↑`jax.block-until-ready` ↓`jaxlib.buffers-wait` |
| Public PJRT execution and completion | `AwaitBuffersReady` | `upstream/jax/jaxlib/util.cc#L56` | IFRT Arrays → ready/status | 一个 array 直接取 GetReadyFuture；多个通过 client 批量取 future；随后等待并取 status。 | ↑`jaxlib.array-wait` ↓`ifrt.array-ready`, `ifrt.values-ready`, `jaxlib.wait-with-signals` |
| Public PJRT execution and completion | `ExecuteShardedOnLocalDevicesInternal` | `upstream/jax/jaxlib/py_executable.cc#L426` | IFRT executable、参数 shards、options → 输出与可选 ShardedToken/status | 检查每个参数的 addressable shard 数；释放 GIL 后调用 IFRT；同一个 IFRT joined status 可填入多个 token，不是再次拆分 native per-device future。 | ↑`jaxlib.execute-sharded` ↓`ifrt.pjrt-execute` |
| Public PJRT execution and completion | `PyLoadedExecutable::ExecuteSharded` | `upstream/jax/jaxlib/py_executable.cc#L485` | PyArray、with_tokens、可选 launch_id → PyExecuteResults | fill_status=with_tokens；执行 stream 缺省取 host thread ID；不得外推 C API/provider 的实际调度。 | ↑`jax.execute-replicated` ↓`jaxlib.execute-local` |
| Public PJRT execution and completion | `PyShardedToken::Await` | `upstream/jax/jaxlib/py_executable.cc#L141` | 一组 future → 等待全部后的合成 status | 本 Python adapter 可把同一个 IFRT joined status 复制到多个 token，不能据 token 个数推导独立设备计时。 | — |
| Public PJRT execution and completion | `PyToken::Await` | `upstream/jax/jaxlib/py_executable.cc#L118` | 有效 execution status future → 等待后的 status | 释放 GIL Await，返回 Python 前展开 UserContext；不同于 output buffer ready。 | — |
| Public PJRT execution and completion | `BlockUntilReadyWithCancel` | `upstream/jax/jaxlib/util.cc#L40` | 一个 C++ Future → host 等待或 Python 信号异常 | 每 200 ms 检查 Python signals；本函数未调用设备 CancelExecution，名称中的 Cancel 不证明设备已取消。 | ↑`jaxlib.buffers-wait` |
| Python API | `matmul` | `upstream/jax/jax/_src/numpy/tensor_contractions.py#L138` | 数组 lhs/rhs、precision、preferred_element_type、out_sharding → 按广播与收缩维度定义的数组 | 检查秩、batch 维度与 sharding；构造 dimension_numbers 后调用 lax.dot_general。 | ↓`jax.dot-general` |
| Python/native boundary | `backend_compile_and_load` | `upstream/jax/jax/_src/compiler.py#L335` | backend Client、MLIR module、设备、CompileOptions、host callbacks → LoadedExecutable 或编译异常 | 真实 backend 与 CompileOnlyPyClient 分支不同；普通路径调用 backend.compile_and_load。 | ↓`jaxlib.compile` |
| Shardy round trip and partitioning | `get_compile_options` | `upstream/jax/jax/_src/compiler.py#L180` | replicas/partitions、device assignment、配置 → CompileOptions | use_spmd_partitioning=True；use_shardy_partitioner 来自 JAX 配置，不代表传播一定发生。 | — |
| Shardy round trip and partitioning | `_to_physical_op_sharding` | `upstream/jax/jax/_src/interpreters/mlir.py#L1203` | aval、sharding 与 axis context → SdyArray 或 OpSharding | 配置决定前端 sharding 表示；extended dtype/manual axes 先处理。 | — |
| Staged API | `Lowered.compile` | `upstream/jax/jax/_src/stages.py#L643` | Lowered 与 compiler_options/device_assignment → Compiled | Lowered.compiler_ir('hlo') 是导出接口，不是逐 native pass 的执行记录。 | — |
| TPU interpreter | `InterpretParams` | `upstream/jax/jax/_src/pallas/mosaic/interpret/params.py#L121` | 随机种子、模拟核心数量、grid recorder 和检查选项 → TPU interpreter 配置 | 本次仅验证单模拟核心和 6 个 grid 点；recorder 接收 token/coordinates/core_id 并返回 token。 | — |
| TPU loader | `make_tpu_client` | `upstream/jax/jax/_src/xla_bridge.py#L198` | libtpu 路径与 client options → TPU C API client | 加载 libtpu.so/PJRT plugin 并注册 profiler；公开接口不提供 libtpu 编译器内部源码。 | — |
| Tracing | `_trace_for_jit` | `upstream/jax/jax/_src/pjit.py#L485` | 函数、JIT 配置、mesh、输入 avals 与参数树 → PjitParams/Jaxpr 及输入输出结构 | Python tracing 与 native 编译分开记录；trace_count 不能单独证明编译次数。 | — |
| Transformation | `grad` | `upstream/jax/jax/_src/api.py#L426` | 标量结果函数与被求导参数位置 → 求梯度的 callable | 通过 value_and_grad 构造结果；本实验使用实数标量 sum(square(matmul))。 | — |
| Transformation | `jit` | `upstream/jax/jax/_src/api.py#L204` | Python callable 与静态参数、donation、sharding、编译选项 → 支持 trace/lower/compile 的 JIT callable | 静态参数与输入抽象类型影响 specialization；调用接口不意味着每次都重编译。 | — |
| Transformation | `vmap` | `upstream/jax/jax/_src/api.py#L1003` | callable、in_axes、out_axes 与轴信息 → 应用 batching 规则的 callable | vmap 是程序变换；本次共同 rhs 的样本形成高秩 dot，不证明执行了 B 次独立 launch。 | — |
| jaxlib binding | `PyClient::CompileAndLoad` | `upstream/jax/jaxlib/py_client.cc#L475` | MLIR module、设备和编译参数 → PyLoadedExecutable | 克隆 module，包装 HloProgram/IFRT 编译选项；这是 JAX 仓库中的 jaxlib 源码。 | ↑`jax.backend-compile` ↓`jaxlib.ifrt-call` |
| jaxlib binding | `PyClient::CompileAndLoadIfrtProgram` | `upstream/jax/jaxlib/py_client.cc#L372` | IFRT Program 与 CompileOptions → IFRT 已加载 executable 的 Python wrapper | 源码在调用 IFRT compiler 和等待 future 时释放 GIL；不据此推断旧挂起案例根因已修复。 | ↑`jaxlib.compile` ↓`ifrt.compile` |

### `stablehlo` — 1 个入口

| 层次 | 符号 | 位置 | 输入 → 输出 | 约束 | 关系 |
|---|---|---|---|---|---|
| StableHLO dialect | `StableHLO_DotGeneralOp` | `upstream/stablehlo/stablehlo/dialect/StablehloOps.td#L2740` | StableHLO dot_general operands 和属性 → 方言定义及验证约束 | 描述操作语义与类型，不指定 CPU/TPU 的机器码或调度。 | — |

### `shardy` — 8 个入口

| 层次 | 符号 | 位置 | 输入 → 输出 | 约束 | 关系 |
|---|---|---|---|---|---|
| Shardy | `addPropagationPipeline` | `upstream/shardy/shardy/dialect/sdy/transforms/propagation/propagation_pipeline.cc#L62` | OpPassManager 与 PropagationOptions → 包含 import/propagation/export 的 pass pipeline | 本次单 CPU 只观测到空 mesh/分片表示；未执行多设备传播验收。 | ↑`xla.shardy-propagate` ↓`shardy.export-pipeline`, `shardy.import-pipeline` |
| Shardy round trip and partitioning | `.Case` | `upstream/shardy/shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.cc#L739` | batch/contracting/noncontracting 维度 → matmul factor rule | batch 保留两边/输出，收缩维度标为 reduction；不能据允许分片推断最优设备划分。 | — |
| Shardy round trip and partitioning | `.Case` | `upstream/shardy/shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.cc#L790` | 向量或矩阵 dot 的输入 shape → factor rule | 本例导回 StableHLO 为 dot；观察到 ([i,k],[k,j])->([i,j])，k 为 reduction。 | — |
| Shardy round trip and partitioning | `addExportPipeline` | `upstream/shardy/shardy/dialect/sdy/transforms/export/export_pipeline.cc#L89` | 传播后 SDY 与 export options → 关闭 sharding、移除辅助信息、可选 reshards/collectives | after_propagation 保存点在清理之后；后续 minimal partitioner 仍使用 global shapes。 | ↑`shardy.propagation` ↓`shardy.minimal-partitioner` |
| Shardy round trip and partitioning | `addImportPipeline` | `upstream/shardy/shardy/dialect/sdy/transforms/import/import_pipeline.cc#L30` | SDY module 与 propagation options → 清理、dataflow edges 与 constraints | before_propagation dump 在部分清理之后，不等于最初输入 module。 | ↑`shardy.propagation` |
| Shardy round trip and partitioning | `runShardyPartitioner` | `upstream/shardy/shardy/dialect/sdy/transforms/export/export_pipeline.cc#L40` | 传播后的 constraints 与 export options → 显式 reshard 或选定 collective/局部 partitioning | 默认 enableInsertExplicitCollectives=false、enablePerInstructionPartitioning=false；本例真正 global→local 在后续 XLA SPMD。 | ↑`shardy.export-pipeline` |
| Shardy round trip and partitioning | `OpPriorityPropagationPassImpl::propagate` | `upstream/shardy/shardy/dialect/sdy/transforms/propagation/op_priority_propagation.cc#L178` | module 与方向规则 → 按 op schedule 的传播 | 配置可直接用 aggressive propagation；源码层次不等于本例执行次数采样。 | ↑`shardy.user-priority` |
| Shardy round trip and partitioning | `UserPriorityPropagationPassImpl::propagate` | `upstream/shardy/shardy/dialect/sdy/transforms/propagation/user_priority_propagation.cc#L231` | sharding references 与用户优先级 → 按优先级运行传播 | 先 priority 0 再逐个用户优先级；本轮未添加不同用户优先级实验。 | ↓`shardy.op-priority` |

### `xla` — 141 个入口

| 层次 | 符号 | 位置 | 输入 → 输出 | 约束 | 关系 |
|---|---|---|---|---|---|
| Analytical GPU model | `AnalyticalLatencyEstimator::NodeCost` | `upstream/xla/xla/service/gpu/model/analytical_latency_estimator.cc#L58` | HloInstruction 与 GPU performance model → 节点模型时间 | 此 NodeCost 没有调用 metadata parser；async collective start/done 使用低成本，其他指令走 performance model。 | — |
| Approximate GPU model | `GpuLatencyEstimator::NodeCost` | `upstream/xla/xla/service/gpu/gpu_latency_hiding_scheduler.cc#L870` | HloInstruction → 估算节点成本 | 先处理 nop；只有 custom-call 分支读取 latency_metadata，其他 opcode 不经此读取。 | ↓`xla.lhs-metadata-parser` |
| Attribute ownership | `ConvertToCustomCall` | `upstream/xla/xla/hlo/transforms/host_offloading_prepare.cc#L102` | host async call 的 inner computation → HostExecute custom call | 复制第一个带任意 frontend attributes 的 inner custom call 后 break；不是汇总多个 kernel 的 latency。 | — |
| Buffer assignment | `BufferAssigner::Run` | `upstream/xla/xla/service/buffer_assignment.cc#L1835` | module、ordering、buffer sizes、alias info、alignment 与 options → BufferAssignment | 分配依赖 schedule 和数据流/alias；静态 allocation 不是进程或设备实测峰值。 | ↑`xla.cpu-buffers` |
| Buffer assignment | `CpuCompiler::CreateBufferAssignment` | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L2459` | 带 schedule 的 HloModule → BufferAssignment | 使用 SequentialHloOrdering、大小函数、alias 信息和对齐要求；估计不等于实测运行峰值。 | ↑`xla.cpu-codegen` ↓`xla.buffer-assignment-entry` |
| Buffer liveness | `BufferAssigner::LiveRangeInterferes` | `upstream/xla/xla/service/buffer_assignment.cc#L1845` | 两个 HloValue 及各自 live ranges → 是否干涉 | 端点相接不自动等于干涉或安全；还需验证 producer/user 能否共享 buffer，copy 有专门限制。 | ↑`xla.buffer-reuse` |
| Buffer reuse | `BufferAssigner::MaybeAssignBuffer` | `upstream/xla/xla/service/buffer_assignment.cc#L1911` | 候选 allocation 与 HloBuffer → 可复用判定及分配 | 检查颜色/空间兼容、容量、只读、live-out、可复用性与存活区间；同一 allocation 的不同 offset 也可能只是打包。 | ↓`xla.buffer-interference` |
| CPU code generation | `CpuCompiler::CompileCpuExecutable` | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L1727` | 优化后的 HloModule 与 IrCompiler/目标选项 → CpuExecutable | 创建 LLVMContext/Module；随后调度、buffer assignment、thunk/IR emission 和编译。 | ↓`xla.cpu-buffers`, `xla.cpu-schedule` |
| CPU codegen preparation | `FusionWrapper::MustWrapInstruction` | `upstream/xla/xla/service/cpu/fusion_wrapper.cc#L29` | 单个 HloInstruction 与 emitter 配置 → 是否包装成 fusion computation | 单算子 wrapper 也有 kFusion；关闭 fusion pass 后仍可能有 wrapper 和库 fusion。 | — |
| CPU collective pipeline | `CpuCompiler::RunHloPassesThroughLayoutAssn` | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L596` | HloModule、AOT 与目标特性 → 布局分配前后的优化 HLO | async-collective pipeline 先重写 custom calls，再用全部为真的谓词将 async collectives 改回同步。 | ↓`xla.async-custom-rewriter`, `xla.async-to-sync` |
| CPU collective policy | `AsyncCollectiveReplacer::RunImpl` | `upstream/xla/xla/hlo/transforms/collectives/async_collective_replacer.cc#L71` | HLO 与 collective predicates → 匹配的 async pair 变为同步 collective | 源码显式删除参与被替换 start/done 的 control_dep calls；不同于实际控制依赖边的保留。 | ↑`xla.cpu-async-pipeline` |
| CPU compiler | `CpuCompiler::RunHloPasses` | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L1188` | HloModule、AOT 标志、目标特性与编译参数 → 被优化的 HLO module | 布局分配前后是不同 pipeline；具体 pass 还受构建、平台与选项控制。 | — |
| CPU compiler | `CpuCompiler::RunHloPassesAfterLayoutAssn` | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L998` | 已有布局的 HloModule 与目标特性 → 规范化、库改写、fusion 等后的 HLO | CpuInstructionFusion、FusionWrapper 与库专用改写不同；HLO 中出现 fusion 不等于业务两算子融合已完成。 | — |
| CPU cost factory | `PjRtCpuClient::GetHloCostAnalysis` | `upstream/xla/xla/pjrt/cpu/cpu_client.cc#L470` | CPU PJRT client → 通用 HloCostAnalysis，使用 CPU ShapeSizeBytes | 没有在此 factory 填入硬件 per_second_rates；不能把 optimal_seconds 缺失当作零耗时。 | ↑`jaxlib.cost-binding` |
| CPU executable and runtime | `PjRtExecutable::CommonMetadata::Serialize` | `upstream/xla/xla/python/pjrt_ifrt/pjrt_executable.cc#L455` | PJRT executable 与 IFRT common metadata → 长度前缀 metadata 加 PJRT opaque payload | IFRT metadata 与后端 executable 是相邻的两段，不能把整体当一个 CPU proto。 | ↓`xla.cpu-executable-serialize` |
| CPU executable and runtime | `CpuAotCompilationResult::Create` | `upstream/xla/xla/service/cpu/cpu_aot_compilation_result.cc#L76` | HLO、buffers、objects、symbols、thunks → CpuAotCompilationResult | 保存实际 thunk sequence 及对象；不是仅记录 HLO 图。 | ↓`xla.cpu-thunk-sequence-proto` |
| CPU executable and runtime | `{` | `upstream/xla/xla/service/cpu/executable.proto#L45` | CPU 编译产物 → CompilationResultProto 字段合同 | 本例 obj_files_kind=KERNELS，含 HLO/config、buffer assignment、thunks、symbols 与 objects。 | — |
| CPU executable and runtime | `DotThunk::Execute` | `upstream/xla/xla/backends/cpu/runtime/dot_thunk.cc#L74` | dot shape/slices 与线程池 → 矩阵乘完成事件 | 从 allocation 取得地址，处理行列布局与 transpose，再调用 TypedMatMul。 | ↑`xla.cpu-thunk-traced-execute` ↓`xla.cpu-eigen-typed` |
| CPU executable and runtime | `DotThunkToProto` | `upstream/xla/xla/backends/cpu/runtime/thunk_serdes/dot_thunk_serdes.cc#L42` | 实际 DotThunk → DotThunkProto | 序列化 dot dimensions、lhs/rhs/output shape 和 allocation slices。 | — |
| CPU executable and runtime | `alignment>` | `upstream/xla/xla/backends/cpu/runtime/dot_lib.h#L53` | 矩阵维度、transpose 与回调 → Eigen contraction 赋值 | 线程池路径与同步无 device 路径不同；不能推导目标 TPU 实现。 | ↑`xla.cpu-eigen-typed` |
| CPU executable and runtime | `TypedMatMul` | `upstream/xla/xla/backends/cpu/runtime/dot_lib.h#L91` | 类型化 lhs/rhs/output 指针与 m/n/k → 对应 alignment 的 MatMul | 按 16-byte 指针对齐选择模板分支；本轮未测实际指针对齐或内部 microkernel。 | ↑`xla.cpu-dot-execute` ↓`xla.cpu-eigen-contract` |
| CPU executable and runtime | `PjRtCpuExecutable::SerializeExecutable` | `upstream/xla/xla/pjrt/cpu/cpu_client.cc#L524` | PjRtCpuExecutable → ExecutableAndOptionsProto | CPU compiler Export 的结果再与 compile options 打包。 | ↑`ifrt.executable-serialize` |
| CPU executable and runtime | `{` | `upstream/xla/xla/backends/cpu/runtime/thunk.proto#L297` | thunk kind/info/impl → oneof 实现与公共身份 | kind 字符串与 oneof 必须一致；DotThunk 与 KernelThunk 分开。 | — |
| CPU executable and runtime | `ThunkSequenceSerDesProtobuf::ToProto` | `upstream/xla/xla/backends/cpu/runtime/thunk_proto_serdes.cc#L928` | ThunkSequence 与资源关系 → ThunkSequenceProto | 逐 thunk 序列化并收集资源使用者；列表顺序本身不能证明并发执行时间线。 | ↑`xla.cpu-aot-create` |
| CPU executable and runtime | `Thunk::TraceMeEncode` | `upstream/xla/xla/backends/cpu/runtime/thunk.cc#L181` | op/module identity、run_id、device ordinal → TraceMe metadata | 导出 JSON 未保留 program_id；end 事件另只带名称，不能任意并发拼接。 | ↑`xla.cpu-thunk-traced-execute` |
| CPU executable and runtime | `ThunkExecutor::TracedExecute` | `upstream/xla/xla/backends/cpu/runtime/thunk_executor.cc#L222` | thunk 与 ExecuteParams → ExecuteEvent 及 producer/consumer trace | TraceMeProducer 围绕 Execute 返回，完成回调生成 end 事件；producer duration 不能一般化为异步完整时长。 | ↓`xla.cpu-dot-execute`, `xla.cpu-thunk-trace-fields` |
| CPU executable and runtime | `YnnFusionThunkToProto` | `upstream/xla/xla/backends/cpu/runtime/thunk_serdes/ynn_fusion_thunk_serdes.cc#L57` | 实际 YnnFusionThunk → YnnFusionThunkProto | 记录 HLO instruction id 和参数/结果 slices；依赖对应 fusion computation。 | — |
| CPU executable introspection | `PjRtCpuExecutable::GetCompiledMemoryStats` | `upstream/xla/xla/pjrt/cpu/cpu_client.cc#L1050` | CPU executable 的 allocation/HLO proto → CompiledMemoryStats 与静态 heap 信息 | 不能用 alias_size 直接证明运行时指针复用；CPU 外部 NumPy view 对照已展示区别。 | — |
| CPU fusion | `CpuInstructionFusion::ShouldFuse` | `upstream/xla/xla/service/cpu/cpu_instruction_fusion.cc#L403` | consumer 与 producer operand index → Allow/Forbid 及原因 | 考虑可发射性、库调用边界、重复计算、concatenate/reduction 限制及公共 fusion 条件；不是数学可合并就必融合。 | ↓`xla.common-fusion-decision` |
| CPU kernel ABI | `Kernel::CallOnce` | `upstream/xla/xla/backends/cpu/runtime/kernel.h#L118` | KernelArg 列表 → 当前线程单 workgroup 调用 | 小 kernel 的 fast path，与多 workgroup 的 Launch 分开定位。 | ↑`xla.kernel-thunk-call` |
| CPU kernel ABI | `Kernel::Launch` | `upstream/xla/xla/backends/cpu/runtime/kernel.cc#L147` | workgroup grid 与 XLA_CPU_KernelArg 列表 → 传入 KernelCallFrame 的实际函数调用 | 同步循环通过 (*kernel_)(&call_frame) 调用；指令反汇编本身不能证明实测性能或 TPU 执行。 | ↑`xla.kernel-thunk-call` |
| CPU pass and dispatch | `AlgebraicSimplifierVisitor::HandleTranspose` | `upstream/xla/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9795` | transpose 及前驱 → 删除恒等转置或改写 dot 等 | 本例还把 transpose(dot) 改写为调换操作数的 dot；不能将整个 algsimp 归结为单一规则。 | — |
| CPU pass and dispatch | `GetDotImplementationStrategy` | `upstream/xla/xla/service/cpu/dot_op_emitter.cc#L1413` | HLO config、dot、target features → DotImplementationStrategy | batch 转 inner dot 后选择策略；形状、布局、类型、配置均影响分支。 | ↑`xla.cpu-dot-thunk` |
| CPU pass and dispatch | `ThunkEmitter::EmitDotThunk` | `upstream/xla/xla/service/cpu/thunk_emitter.cc#L985` | dot、buffer assignment、target features → KernelThunk 或 DotThunk | 由 implementation strategy 分支决定；不能仅凭没有 .o 推断具体分支。 | ↓`xla.cpu-dot-strategy` |
| CPU pass and dispatch | `CpuLayoutAssignment::AddBackendConstraints` | `upstream/xla/xla/service/cpu/cpu_layout_assignment.cc#L133` | CPU HLO 与 layout constraints → operand/result 布局约束 | 与通用 layout assignment 配合；本例出现 copy，不等价于运行流量测量。 | — |
| CPU pass and dispatch | `YnnFusionThunk::YnnExecutable::Invoke` | `upstream/xla/xla/backends/cpu/runtime/ynnpack/ynn_fusion_thunk.cc#L90` | 线程池、参数和结果地址 → ynn_invoke_runtime 状态 | 设置 external values 与线程池后调用 YNN runtime；不推导库内部机器码。 | — |
| CPU pass and dispatch | `ThunkEmitter::EmitYnnFusionThunk` | `upstream/xla/xla/service/cpu/thunk_emitter.cc#L1410` | YNN fusion 与 allocation slices → YNN subgraph builder / thunk | 收集参数与结果，常量单独捕获；源码路径不是本次 runtime sampling。 | — |
| CPU pass and dispatch | `CanonicalizeOperand` | `upstream/xla/xla/hlo/transforms/expanders/dot_decomposer.cc#L68` | dot operand 与 batch/contracting 维度 → transpose 和二维/三维 reshape | 合并非收缩维度；本例共享 W 的 batch 展平成 12 行。 | — |
| CPU pass and dispatch | `IdentityReshapeRemoving` | `upstream/xla/xla/hlo/transforms/simplifiers/dynamic_dimension_simplifier.cc#L158` | reshape 与其 operand → operand use 重接 | Shape::Equal 才转发；此函数没有立即删除原 reshape。 | — |
| CPU pass and dispatch | `CreateLibraryFusion` | `upstream/xla/xla/backends/cpu/transforms/library_rewriter.cc#L66` | 库 matcher 选中的 HLO 指令 → kCustom fusion 与 backend_config | 写入库 fusion kind 并替换原指令；本例为 __ynn_fusion。 | — |
| CPU pass and dispatch | `HandleReshape` | `upstream/xla/xla/hlo/transforms/expanders/reshape_decomposer.cc#L34` | 带 layout 的 reshape → bitcast 或 copy/bitcast | 只有 layout 兼容才只需 bitcast；其他分支可插入一个或两个 copy。 | — |
| CPU pass and dispatch | `ReshapeMover::RunImpl` | `upstream/xla/xla/hlo/transforms/simplifiers/reshape_mover.cc#L399` | 非 fusion computation 的候选 → 移过 elementwise 的 reshape | 本例乘 2 从 [3,4,6] 改为 [12,6]；后续 algsimp 清理连锁 reshape。 | — |
| CPU pass and dispatch | `TransposeFolding::RunImpl` | `upstream/xla/xla/service/transpose_folding.cc#L221` | dot/convolution 的转置操作数 → 可由后端接受的维度改写 | 先调用后端合法性回调；本例 W.T 折入 rhs contracting dims。 | — |
| CPU runtime invocation | `num_results>::ExecuteInternal` | `upstream/xla/xla/backends/cpu/runtime/kernel_thunk.cc#L166` | BufferAllocations、FunctionLibrary、threadpool → Kernel 调用完成事件 | 从 tensor slices 取得地址，按名字解析 kernel function；单 workgroup、线程池与同步循环为不同路径。 | ↓`xla.kernel-call-once`, `xla.kernel-launch` |
| CPU scheduling | `CpuCompiler::CreateHloSchedule` | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L2441` | HloModule → HloSchedule | 源码按配置选择 DFS memory scheduler 或 BFS scheduler；不代表 TPU 调度实现。 | ↑`xla.cpu-codegen` |
| Collective policy | `AsyncCollectiveCreator` | `upstream/xla/xla/hlo/transforms/collectives/async_collective_creator.h#L38` | collective predicates 与阈值配置 → 同步 collective 的 async start/done 表达 | 各后端配置决定是否转换；generic 模式默认 false，不能把该声明当作 libtpu 实际配置。 | — |
| Collective representation | `AsyncCollectiveCustomCallRewriter::RunImpl` | `upstream/xla/xla/service/async_collective_custom_call_rewriter.cc#L467` | collective start/done custom calls → 原生 async HLO 表达 | 识别匹配 start/done 与数据流路径；这是表示重写，不保证异步执行。 | ↑`xla.cpu-async-pipeline` |
| Collective rewrite ownership | `FinishRewrite` | `upstream/xla/xla/service/async_collective_custom_call_rewriter.cc#L83` | 旧 custom calls、新 async pair 与 forward path → 更新 uses/control edges 并清理旧指令 | 固定源码删除 start 前检查 user_count==0；当前 wheel 的 live-start 删除错误不能据此直接归因于同版 C++。 | — |
| Compiler profiling | `HloPassPipeline::RunPassesInternal` | `upstream/xla/xla/hlo/pass/hlo_pass_pipeline.cc#L141` | HloModule、debug options、execution threads 和 pass 列表 → 每个 pass 的 TraceMe 范围与 metadata/dump | TraceMe 在过滤 gate 前创建，范围也包含 metadata/invariant/dump 开销；事件名称出现不单独证明 pass 改变了 IR。 | — |
| Control dependencies | `ControlDepRewriter::RunImpl` | `upstream/xla/xla/service/control_dep_rewriter.cc#L32` | target=control_dep 的两个 operands → src→dst 控制依赖并删除 custom call | CPU sync-control capture 已在 pass 前后验证 dot→all-reduce 边；不证明 overlap。 | — |
| Cost analysis | `HloCostAnalysis::HandleCustomCall` | `upstream/xla/xla/service/hlo_cost_analysis.cc#L1362` | 未知 custom-call instruction → 未知 cost 的 -1 sentinel（内部 call markers 例外为 0） | CPU analyzer 对本次 tpu_custom_call 返回 -1；这不是 libtpu backend cost 行为的验证。 | — |
| Cost analysis | `HloCostAnalysis::Postprocess` | `upstream/xla/xla/service/hlo_cost_analysis.cc#L91` | 当前 properties、per_second_rates 与最小延迟选项 → 逐指令 bottleneck time 与累计 properties | 最大资源时间需要配置 rates；默认空 rates 不产生目标硬件 roofline 或真实运行时间。 | — |
| Cost analysis | `HloCostAnalysis::Preprocess` | `upstream/xla/xla/service/hlo_cost_analysis.cc#L58` | HloInstruction 与 shapes → 默认输入/输出字节、utilization 等 properties | 这是静态估计，后续 handler 可覆盖，不等于实际 HBM/DRAM 流量。 | — |
| Cost analysis | `HloCostAnalysis::GetDotFlops` | `upstream/xla/xla/service/hlo_cost_analysis.cc#L484` | lhs/result shape 与 DotDimensionNumbers → 按 FMA 口径计数的 FLOPs | 静态算量估计需要硬件带宽/吞吐和流量假设才能构成 roofline。 | — |
| Fusion legality | `InstructionFusion::ShouldFuse` | `upstream/xla/xla/service/instruction_fusion.cc#L1133` | consumer 与 producer operand index → 公共 fusion 合法性/启发式判断 | CPU 的专用条件仍需额外满足；调用本层不能保证最终形成目标 fusion。 | ↑`xla.cpu-fusion-decision` |
| GPU scheduling gate | `IsLHSEnabled` | `upstream/xla/xla/service/gpu/gpu_hlo_schedule.cc#L863` | module options、fingerprint 与 GPU description → 是否启用 LHS | 显式 flag 优先；还看 opt effort、SOL 支持与有效 PGLE profile。标记本身不绕过 gate。 | — |
| GPU scheduling model selection | `GetLatencyEstimator` | `upstream/xla/xla/service/gpu/gpu_hlo_schedule.cc#L491` | module、GPU description、fingerprint、config → PGLE/analytical/SOL/approximate estimator | profile 优先；其后 analytical flag、受支持的 SOL 与 fallback。选择依赖真实配置和设备，本轮只阅读源码。 | ↑`xla.gpu-lhs-pipeline` |
| GPU scheduling pipeline | `RunLatencyHidingSchedulerPasses` | `upstream/xla/xla/service/gpu/gpu_hlo_schedule.cc#L680` | HLO、GPU info、memory limit、config → LHS 及相关标记/检查后的 module | 把选定 estimator 放入 SchedulingContext，并 AddPass<LatencyHidingScheduler>；不能据此声称本次 CPU 调用了该管线。 | ↓`xla.gpu-estimator-select`, `xla.lhs-pass` |
| IFRT | `PjRtCompiler::CompileAndLoad` | `upstream/xla/xla/python/pjrt_ifrt/pjrt_compiler.cc#L91` | Program 与 IFRT CompileOptions → LoadedExecutableRef future | 这个实现要求 HloProgram，翻译设备 ID，再创建 PjRtLoadedExecutable。 | ↑`jaxlib.ifrt-call` ↓`ifrt.pjrt-create` |
| IFRT/PJRT | `PjRtLoadedExecutable::Create` | `upstream/xla/xla/python/pjrt_ifrt/pjrt_executable.cc#L744` | MLIR module、PJRT client、编译选项与设备 → 包装的 PJRT executable | 调用 client->pjrt_client()->CompileAndLoad；调用前读取捐赠、shape/layout 等 metadata。 | ↑`ifrt.compile` ↓`pjrt.c-api-compile` |
| JIT code memory | `ContiguousSectionMemoryManager::finalizeMemory` | `upstream/xla/xla/backends/cpu/codegen/contiguous_section_memory_manager.cc#L157` | JIT code 与只读 data memory blocks → 代码页权限与指令缓存更新 | 代码区设置 READ/EXEC，只读区设置 READ；这里不是 tensor 的 buffer allocation 或 TPU VMEM。 | — |
| JIT object linking | `CreateObjectLinkingLayer` | `upstream/xla/xla/backends/cpu/codegen/execution_engine.cc#L37` | ExecutionSession → RTDyldObjectLinkingLayer | 此 pin 的 XLA 使用 RTDyld 层和 ContiguousSectionMemoryManager，不能仅凭 ORC 名称称其使用 JITLink。 | — |
| LLVM compiler adapter | `IrCompiler::operator` | `upstream/xla/xla/backends/cpu/codegen/ir_compiler.cc#L289` | llvm::Module → 机器码对象 MemoryBuffer 或错误 | 串联 target machine、IR hooks、RunIrPasses、EmitMachineCode；并发编译需独立 TargetMachine。 | ↑`llvm.orc-ir-emit` ↓`xla.emit-object`, `xla.llvm-pass-manager` |
| LLVM observation | `GetIRModuleHooks` | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L1245` | HLO module 与 user hooks → LLVM 优化前后 hooks | ir-no-opt 是本段 LLVM pipeline 前的观察点，不是原始 Jaxpr 或未经任何 lowering 的程序。 | — |
| LLVM optimization | `IrCompiler::RunIrPasses` | `upstream/xla/xla/backends/cpu/codegen/ir_compiler.cc#L354` | LLVM module、TargetMachine 与 options → 优化后的 module | 使用新 pass manager，按 O0/其他选择 pipeline；IPO、vectorization 与代码生成不能混成单个 HLO pass。 | ↑`xla.ir-compile` ↓`llvm.default-pipeline` |
| Latency hiding scheduling | `SchedulerConfig` | `upstream/xla/xla/service/latency_hiding_scheduler.h#L143` | overlap limits、memory limit、资源与策略选项 → scheduler 配置 | 通用开源接口；本次 CPU 走 DFS/BFS，未证明目标 TPU 使用此配置。 | — |
| Latency hiding scheduling | `LatencyHidingScheduler` | `upstream/xla/xla/service/latency_hiding_scheduler.h#L2176` | SchedulingContext、SchedulerCore 与 module → 调度与模型统计 | 存在源码类不代表 CPU 或 TPU pipeline 调用了它；目标部署必须另行取证。 | ↑`xla.gpu-lhs-pipeline` |
| Latency metadata | `LatencyEstimator::GetLatencyFromMetadata` | `upstream/xla/xla/service/latency_hiding_scheduler.cc#L376` | HLO frontend_attributes[latency_metadata] → optional TimeCost：int64 ns × CyclesPerMicrosecond / 1000 | 缺失或 SimpleAtoi 失败返回 nullopt；原生 CPU 单元测试已确认零/负数接受、int64 越界拒绝和配置的单位缩放，目标 GPU/TPU 消费另行取证。 | ↑`xla.gpu-node-cost`, `xla.sol-node-cost` |
| Liveness analysis | `HloLiveRange::Run` | `upstream/xla/xla/hlo/utils/hlo_live_range.cc#L53` | HloSchedule、alias analysis、computation → 以 instruction order 为逻辑时钟的 live ranges | 逻辑时间不是微秒；输出和输入的范围、alias normalization 与 physical allocation 要分开解释。 | — |
| MLIR/HLO boundary | `MlirToXlaComputation` | `upstream/xla/xla/pjrt/mlir_to_hlo.cc#L99` | MLIR ModuleOp 与 tuple/shardy 编译选项 → XlaComputation/HLO | 先处理 CHLO、常量作用域及分片等转换，再导出；不是只更换文本格式。 | ↓`xla.gspmd-detection`, `xla.sdy-export-for-gspmd`, `xla.sdy-roundtrip-export` |
| MLIR/HLO boundary | `ConvertStablehloToHloProtoInternal` | `upstream/xla/xla/hlo/translate/stablehlo.cc#L103` | MLIR module 与导出选项 → HloProto | 当前代码设置 direct_stablehlo_to_hlo=true；不能假设所有操作都先完整改写为 MHLO。 | — |
| Machine code emission | `IrCompiler::EmitMachineCode` | `upstream/xla/xla/backends/cpu/codegen/ir_compiler.cc#L477` | LLVM module 与 target machine → 包含可重定位对象的 SmallVectorMemoryBuffer | 该阶段使用 legacy codegen PassManager 和 addPassesToEmitMC；与前面的 LLVM IR pass manager 不同。 | ↑`xla.ir-compile` ↓`llvm.mc-emission` |
| Memory diagnostics | `HloLiveRange::ComputePeakMemoryMoment` | `upstream/xla/xla/hlo/utils/hlo_live_range.cc#L325` | buffer live ranges 和 HLO shapes → 诊断 peak moment | 此算法同一时刻先处理 start 再处理 end+1；本次捕获的 reduce 诊断选点不是打印闭区间总量的最大值，需复验匹配 binary。 | — |
| Metadata export | `CreateFrontendAttributes` | `upstream/xla/xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc#L970` | MLIR 命名属性数组 → XLA FrontendAttributes 字符串 map | 此重载接受 StringAttr 和 BoolAttr；直接写入 IntegerAttr 的自定义值未被导出。 | — |
| Mosaic TPU dialect | `TPU_TraceStartOp` | `upstream/xla/xla/mosaic/dialect/tpu/tpu_ops.td#L1618` | message 字符串和 level i32 属性 → 无 SSA 结果的 trace_start 操作 | 方言定义不是设备时钟或采样实现；Mosaic TPU MLIR 不是 LLO。 | — |
| Mosaic TPU dialect | `TPU_TraceStopOp` | `upstream/xla/xla/mosaic/dialect/tpu/tpu_ops.td#L1623` | 无显式输入 → 无 SSA 结果的 trace_stop 操作 | 配对需要检查 IR 的嵌套和控制流；本次仅验证直线 kernel。 | — |
| Native HLO Python API | `set_frontend_attribute` | `upstream/xla/xla/python/hlo.cc#L783` | HloInstruction wrapper、string key/value → 更新 instruction 的 FrontendAttributes map | 可修改 native HLO 属性；语义性图编辑仍受 shape/opcode/effect/alias 等验证约束。 | — |
| ORC materialization | `JitCompiler::Compile` | `upstream/xla/xla/backends/cpu/codegen/jit_compiler.cc#L193` | 带类型标识的编译符号列表 → FunctionLibrary | 通过 ObjectLoader lookup 触发 materialization，并等待派发任务结束；源码中的旧 SimpleOrcJIT 注释不能代替当前调用。 | ↓`xla.compiled-function-library`, `xla.object-lookup` |
| ORC registration | `JitCompiler::AddModule` | `upstream/xla/xla/backends/cpu/codegen/jit_compiler.cc#L173` | ThreadSafeModule、dylib_index → 添加到 IRCompileLayer 的 module | 设置 target triple/data layout/xla_dylib_index；AddModule 不等于已完成全部符号 materialization。 | — |
| ORC symbol resolution | `ObjectLoader::LookupSymbols` | `upstream/xla/xla/backends/cpu/codegen/object_loader.cc#L172` | symbols 与 JITDylibs → SymbolMap 或 unresolved-symbol 错误 | 按 data layout 修饰名字，只查 exported symbols；不是任意字符串直接当函数地址。 | ↑`xla.jit-compile` ↓`llvm.orc-lookup` |
| Object observation | `CreateOrcJITPostCompilationHook` | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L1406` | LLVM module、ObjectFile 与 HLO module → ObjFileProto 留存和可选 .o dump | 对象字节另存入 executable，dump 并不是通过系统链接器构造独立 .so。 | — |
| PJRT C API | `PjRtCApiClient::CompileAndLoad` | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L763` | MLIR module 与 CompileOptions → PjRtLoadedExecutable | 经 InitializeArgsAndCompile 到 PJRT_Client_Compile；plugin 内部实现另行取证。 | ↑`ifrt.pjrt-create` |
| PJRT ownership | `CommonPjRtBuffer::GetBufferForDonationHoldLocked` | `upstream/xla/xla/pjrt/abstract_tracked_device_buffer.cc#L233` | buffer 状态与 usage/external/donation holds → 取得 donation storage 或返回错误 | 固定公共实现拒绝已有外部引用的 donation；当前 CPU wheel 的具体 fallback 路径仍受 VERSION-SKEW 限制。 | — |
| Pass infrastructure | `HloPassPipeline::RunPassesInternal` | `upstream/xla/xla/hlo/pass/hlo_pass_pipeline.cc#L141` | HLO、DebugOptions、execution_threads → changed 标志与被修改的 HLO | 字面量 dump regex .* 特判跳过未变化的 pass；.+ 可避免这项特判。 | — |
| Profile-guided model | `ProfileGuidedLatencyEstimator::NodeCost` | `upstream/xla/xla/service/profile_guided_latency_estimator.cc#L137` | HloInstruction、instruction profile map、fallback estimator → profile cost 或 fallback cost | async collective start/done 的低成本优先，其后按指令名命中 profile；只有缺失时调用 fallback NodeCost，metadata 不是无条件最高优先级。 | — |
| Profiler schema | `IsInternalStat` | `upstream/xla/xla/tsl/profiler/utils/xplane_schema.cc#L615` | 可选 StatType → 该字段是否属于不向 trace viewer 导出的内部 stat | _pt/_p/_ct/_c 与 program_id 均为 internal；未知自定义字段可保留。编译 Hack 另使用 research_program_id 供导出分组。 | ↑`xla.xplane-trace-events` |
| Public PJRT execution and completion | `PjRtArray::GetReadyFuture` | `upstream/xla/xla/python/pjrt_ifrt/pjrt_array.cc#L616` | 一个 IFRT Array 的本地 PJRT buffers → 一个或 JoinFutures 的 ready future | 只覆盖该 Array 所持 buffers，不是全设备/全进程 barrier。 | ↑`jaxlib.buffers-wait` ↓`pjrt.buffer-ready-future` |
| Public PJRT execution and completion | `Execute` | `upstream/xla/xla/python/ifrt/executable.h#L317` | 设备上的输入 Arrays、options、可选 devices → ExecuteResult | 声明附近明确没有严格 barrier 语义；输入可用与输出 ready 的粒度由 backend 决定。 | — |
| Public PJRT execution and completion | `{` | `upstream/xla/xla/python/ifrt/executable.h#L125` | launch/donation/stream/status 配置 → IFRT 运行选项 | fill_status 只控制 ExecuteResult.status 的有效性；execution_stream_id 的接口合同不能证明每种 adapter 都完整转发。 | — |
| Public PJRT execution and completion | `PjRtLoadedExecutable::Execute` | `upstream/xla/xla/python/pjrt_ifrt/pjrt_executable.cc#L840` | PjRtCompatibleArray 列表与选项 → IFRT Array 输出及可选 joined status | 非 portable 分支始终请求 PJRT futures 并 JoinFutures，fill_status 只控制最终返回字段；不是对 TPU runtime 的实测。 | ↑`jax.pjit-fast-execute`, `jaxlib.execute-local` ↓`pjrt.capi-execute` |
| Public PJRT execution and completion | `PjRtClient::GetReadyFuture` | `upstream/xla/xla/python/pjrt_ifrt/pjrt_client.cc#L1930` | 若干 IFRT Value → JoinFutures | 批量等待针对传入 values，不凭空加入其他计算或其他 host。 | ↑`jaxlib.buffers-wait` |
| Public PJRT execution and completion | `{` | `upstream/xla/xla/pjrt/c/pjrt_c_api.h#L2641` | PJRT_Buffer → caller 拥有的 ready event | 数据 ready 或错误均触发；删除/捐赠与先取得 event 的时序有单独合同，不等同整个 executable 完成。 | ↑`pjrt.buffer-ready-event` |
| Public PJRT execution and completion | `PjRtCApiBuffer::GetReadyEvent` | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L3993` | buffer → 缓存的 PJRT_Buffer_ReadyEvent | readiness_event_ 管理 C event 的寿命；与执行完成事件是不同 API 入口。 | ↑`pjrt.buffer-track-event` ↓`pjrt.buffer-ready-contract` |
| Public PJRT execution and completion | `PjRtCApiBuffer::GetReadyFuture` | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L4060` | buffer 的删除状态与 readiness cache → 缓存的 C++ future 或错误 | 删除/捐赠先拒绝；mutex 下只建立一次 promise，ready 不等同 status.ok。 | ↑`ifrt.array-ready` ↓`pjrt.buffer-track-event` |
| Public PJRT execution and completion | `PjRtCApiBuffer::MakePromiseTrackEvent` | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L4006` | readiness promise 与 event → ready/error promise | 先 IsReady，已就绪则 Error 并立即 Set，否则 OnReady；不直接调用 PJRT_Event_Await。 | ↑`pjrt.buffer-ready-future` ↓`pjrt.buffer-ready-event`, `pjrt.event-callback-contract`, `pjrt.event-ready-contract` |
| Public PJRT execution and completion | `{` | `upstream/xla/xla/pjrt/c/pjrt_c_api.h#L2051` | 本地设备×参数列表、输出存储、可选完成事件 → buffer handles 与每设备完成事件 | 输出列表由 caller 分配；输出 buffer/event 由 caller Destroy；调用直接报错时完成事件不会被填充。 | ↑`pjrt.capi-execute` |
| Public PJRT execution and completion | `{` | `upstream/xla/xla/pjrt/c/pjrt_c_api.h#L1992` | 调用选项与回调地址 → C ABI 字段 | 此 pin 没有 execution_stream_id 字段；callback 函数寿命超过提交调用，call_location 指针只需覆盖调用。 | — |
| Public PJRT execution and completion | `PJRT_LoadedExecutable_Execute` | `upstream/xla/xla/pjrt/c/pjrt_c_api_wrapper_impl.cc#L2313` | C Execute Args → 调用内层 C++ PJRT provider 后包装结果 | 开源参考 adapter，仅 SOURCE-ONLY；不能无证据把它放进 libtpu.so 内部调用栈。 | ↓`pjrt.profiler-context` |
| Public PJRT execution and completion | `PjRtCApiLoadedExecutable::Execute` | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L3412` | C++ inputs、options、可选 returned_futures → C++ 输出 buffer 包装与 future | 经函数指针进入 plugin；本索引未证明 libtpu 使用开源 wrapper_impl；库调用 TraceMe 不是设备事件。 | ↑`ifrt.pjrt-execute` ↓`pjrt.c-execute-args`, `pjrt.capi-execute-args`, `pjrt.event-to-future` |
| Public PJRT execution and completion | `PjRtCApiLoadedExecutable::GetCommonExecuteArgs` | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L3273` | C++ PJRT buffers、options 与 backing storage → PJRT_LoadedExecutable_Execute_Args | 映射 launch/non-donatable/callback 等；字段版本 gate 和 storage 生存期必须保留。 | ↑`pjrt.capi-execute` |
| Public PJRT execution and completion | `{` | `upstream/xla/xla/pjrt/c/pjrt_c_api.h#L368` | event、callback、user_arg → 注册 ready callback 的参数 | 回调接收并负责销毁 PJRT_Error；user_arg 所有权仍在 caller，不指定设备或回调执行线程。 | ↑`pjrt.buffer-track-event`, `pjrt.event-to-future` |
| Public PJRT execution and completion | `PJRT_Event_IsReady` | `upstream/xla/xla/pjrt/c/pjrt_c_api.h#L330` | PJRT_Event → is_ready | ready 包含错误完成；必须检查 Error 或 future status，不能等同数值成功。 | ↑`pjrt.buffer-track-event` |
| Public PJRT execution and completion | `ConvertCEventToCppFuture` | `upstream/xla/xla/pjrt/c/pjrt_c_api_helpers.cc#L374` | 拥有的 PJRT_Event 与 API 表 → C++ future | 通过 OnReady 回调置 promise 并释放事件；注册失败分支另返错误 future，不能据成功分支宣称所有错误路径资源处理已验证。 | ↑`pjrt.capi-execute` ↓`pjrt.event-callback-contract` |
| Public PJRT execution and completion | `GetTracemeContextId` | `upstream/xla/xla/pjrt/c/pjrt_c_api_helpers.h#L328` | API 或 Args 的 extension 链 → context ID，缺失时 -1 | 调用方读取 API extension 与传给 plugin 的 Args extension 必须区分；缺失 ID 不能当作唯一关联。 | ↑`pjrt.c-wrapper-execute` |
| Public PJRT execution and completion | `CreatePjrtProfilerExtension` | `upstream/xla/xla/pjrt/c/pjrt_c_api_helpers.cc#L1055` | 库调用 trace 名称 → 带 context ID 的 Profiler extension | 短生命周期 TraceMeProducer 使用 kPjrtLibraryCall=14；不是 kernel 区间，也未由此前仅支持 0/15 的 matcher 验证。 | ↑`pjrt.capi-execute` |
| Python/native profiling | `TraceMeWrapper` | `upstream/xla/xla/python/profiler.cc#L59` | Python name/kwargs → 拥有原生 tsl::profiler::TraceMe 的 wrapper | 这里是实际 nanobind 注册使用的 wrapper；构造启动，__enter__ 返回自身，__exit__ 调用 Stop。 | — |
| Runtime function binding | `ObjectLoader::CreateFunctionLibrary` | `upstream/xla/xla/backends/cpu/codegen/object_loader.cc#L206` | 请求的 symbols 与解析后的 SymbolMap → 持有 ExecutionEngine 与已解析地址的函数库 | 函数库持有引擎生命周期；地址与 tensor BufferAssignment 是不同对象。 | ↑`xla.jit-compile` |
| Scheduling cost interface | `LatencyEstimator` | `upstream/xla/xla/service/latency_hiding_scheduler.h#L198` | HLO node 与依赖边 → NodeCost、GetLatencyBetween 与时钟换算 | 模型预测不是实测通信时间；latency_metadata 的两处非测试调用位于 GPU NodeCost 实现，不能推广为 TPU 或默认 cost_analysis 消费。 | — |
| Scheduling graph | `HloScheduleGraph::HloScheduleGraph` | `upstream/xla/xla/service/latency_hiding_scheduler.cc#L2938` | 原 instruction order、SchedulingContext 与 reachability → 带 node cost、edge latency 和资源信息的图 | NodeCost 写入每个 HloGraphNode；这些是模型值，不是 trace 事件时间。 | — |
| Scheduling model clock | `DefaultSchedulerCore::ScheduleNode` | `upstream/xla/xla/service/latency_hiding_scheduler.cc#L2582` | 候选 node 与调度状态 → 更新模型 ready time 和当前时间 | schedule_time 受依赖 edge latency 与资源限制，current_time=schedule_time+node cost；不是直接按标签排序。 | — |
| Shardy round trip and partitioning | `CpuCompiler::RunHloPassesThroughLayoutAssn` | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L596` | HLO config 与 CPU options → 分片传播/分区或单分区清理 | num_partitions>1 时传播再 StatefulRngSpmdPartitioner；单分区构造 ShardyXLA(false)。 | ↓`xla.shardy-pass` |
| Shardy round trip and partitioning | `hasGspmdAttrsOrOps` | `upstream/xla/xla/service/spmd/shardy/utils.cc#L360` | 可能混有旧式 sharding 的 module → 是否触发 GSPMD 兼容判断 | 非 main 参数、func results 和 Sharding custom calls 等有具体条件；不以任意 mhlo.sharding 存在就回退。 | ↑`xla.mlir-to-hlo` |
| Shardy round trip and partitioning | `SdyRoundTripExportShardyAttrsPass` | `upstream/xla/xla/service/spmd/shardy/sdy_round_trip/export_shardy_attrs.cc#L150` | SDY sharding/rules/meshes → 隐藏 frontend attrs 或 V3 表示 | V3 开启时 frontend attrs 只存 rules；本轮 CPU 样本观察到非 V3 的 xla.sdy.meshes/sharding。 | — |
| Shardy round trip and partitioning | `SdyRoundTripImportShardyAttrsPass` | `upstream/xla/xla/service/spmd/shardy/sdy_round_trip/import_shardy_attrs.cc#L575` | 隐藏 attrs 与 V3 开关 → 恢复 mesh/tuple shardings 与 op attributes | 恢复后的表示不能据函数名认为图或全部 HLO metadata 字节不变。 | — |
| Shardy round trip and partitioning | `ExportShardyForGSPMD` | `upstream/xla/xla/pjrt/mlir_to_hlo.cc#L234` | 含 Shardy mesh 的 module → 保留约束的 StableHLO | 无 mesh 提前返回；用于 GSPMD 兼容，区别于保存 SDY 信息的 round-trip export。 | ↑`xla.mlir-to-hlo` |
| Shardy round trip and partitioning | `addStablehloExportPipeline` | `upstream/xla/xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_export.cc#L31` | 传播后 SDY module 和 export options → 可供后续 HLO/SPMD 的 StableHLO | sdy.constant→stablehlo.constant、ops、shard_map、shardings、callbacks；不是最初保存 round-trip 信息的 export。 | ↑`xla.shardy-propagate` |
| Shardy round trip and partitioning | `addSdyRoundTripExportPipeline` | `upstream/xla/xla/service/spmd/shardy/sdy_round_trip/pipelines.cc#L52` | SDY/StableHLO、mesh/V3 选项 → 可转 HLO 且保存 SDY 信息的 pipeline | lift/dedup、ops、shard_map、attrs 和 StableHLO shardings 各有步骤。 | ↑`xla.mlir-to-hlo` |
| Shardy round trip and partitioning | `addSdyRoundTripImportPipeline` | `upstream/xla/xla/service/spmd/shardy/sdy_round_trip/pipelines.cc#L72` | HLO 导回的 StableHLO 与隐藏属性 → 恢复 SDY 的 pipeline | 常量转为 sdy.constant 防止后续折叠；恢复 mesh、attrs、custom calls、shard_map。 | ↑`xla.shardy-propagate` |
| Shardy round trip and partitioning | `getShardyDirIfShouldDump` | `upstream/xla/xla/service/spmd/shardy/shardy_xla_pass.cc#L304` | dump 目录、pass regex、verbose → Shardy dump 根目录或空串 | 需 xla_dump_to 与匹配 pass；详细 Shardy 文件由内部保存点产生，不等于所有 pass 逐项 dump。 | ↑`xla.shardy-propagate` |
| Shardy round trip and partitioning | `ShardyXLA::RunImpl` | `upstream/xla/xla/service/spmd/shardy/shardy_xla_pass.cc#L470` | HloModule、propagation 选项 → 替换 computation 后的 HLO | 可提前返回；执行往返时保存/恢复 entry layout、alias、donor config。非完整模块直接替换。 | ↑`xla.cpu-spmd-gate` ↓`xla.shardy-propagate`, `xla.shardy-replace-computations` |
| Shardy round trip and partitioning | `runShardingPropagation` | `upstream/xla/xla/service/spmd/shardy/shardy_xla_pass.cc#L318` | 从 HLO 导入的 MLIR 与 options → SDY import/propagation/export 结果 | production 使用 SDY round-trip import；importMhloShardings 分支仅测试。V3 与旧 frontend payload 有独立 fallback。 | ↑`xla.shardy-pass` ↓`shardy.propagation`, `xla.sdy-final-export`, `xla.sdy-roundtrip-import` |
| Shardy round trip and partitioning | `createFromProtoAndReplaceComputations` | `upstream/xla/xla/service/spmd/shardy/shardy_xla_pass.cc#L97` | 转换后的 HloModuleProto 与原 HloModule → 重建/替换 computations 并 DCE | 名称/ID 可重新分配；不能用逐字文本相等作为往返语义保留的唯一判断。 | ↑`xla.shardy-pass` |
| Trace export | `AddTraceEvent` | `upstream/xla/xla/tsl/profiler/convert/trace_events_to_json.cc#L87` | TraceEvent 的 ps 时间、device/resource ID、名称和 args → Chrome Trace Event JSON 的 X 事件 | ts/dur 转成微秒，raw duration=0 时先夹到 1 ps；displayTimeUnit=ns 不改变数值单位。嵌套时长不可直接当 wall time。 | — |
| Trace export | `ConvertXPlaneToTraceEvents` | `upstream/xla/xla/tsl/profiler/convert/xplane_to_trace_events.cc#L62` | XPlaneVisitor 的事件、共享 metadata 与 occurrence stats → TraceContainer 中的 TraceEvent 及可见 args | 过滤 internal context/program 字段；同名 stat 后值覆盖前值。raw XLine 的完整 ID 在这里转成 uint32 viewer resource ID。 | ↓`xla.internal-trace-stat` |
| Unified GPU model | `SolLatencyEstimator::NodeCost` | `upstream/xla/xla/service/gpu/model/sol_latency_estimator.cc#L521` | HloInstruction、GPU model/table config → 估算节点成本 | 任何 opcode 都先尝试 latency_metadata，再判断 async、matmul、fusion 等分支；不能把该优先级套到其他 estimator。 | ↓`xla.lhs-metadata-parser` |
| Upstream test specification | `TEST` | `upstream/xla/xla/service/latency_hiding_scheduler_test.cc#L158` | 上游构造的 custom call：1000、invalid、missing → 测试规范中的 optional time 断言 | 该固定测试已在 CPU 原生目标运行通过；独立测试补丁另补 14 个边界样本。测试配置的 cycles/us 不等于硬件频率。 | — |
| XSpace contexts and export | `AddFlowsToXplane` | `upstream/xla/xla/tsl/profiler/utils/xplane_utils.cc#L497` | host_id、connect_traceme 与 XPlane → 由 context/correlation 生成的 flow stats | 需要显式开启 connect_traceme；本轮自定义 JSON overlay 没有调用此原生方法。 | ↑`xprof.context-preprocess` |
| XSpace contexts and export | `ConnectContextGroups` | `upstream/xla/xla/tsl/profiler/utils/group_events.cc#L200` | ContextGroupMap → producer 到 consumer 的有向关联 | 原生实现支持组内多对多；本轮应用合同只接受一对一且不猜测歧义。 | — |
| XSpace contexts and export | `SetContextGroup` | `upstream/xla/xla/tsl/profiler/utils/group_events.cc#L176` | GroupingEventStats 与 EventNode → 按 type/id/PID 分组的两类节点 | 使用 optional.has_value，不能因 ID 或 type 为零而丢弃。 | — |
| XSpace contexts and export | `GroupingEventStats::GroupingEventStats` | `upstream/xla/xla/tsl/profiler/utils/group_events.cc#L103` | 单个事件的 occurrence stats → producer/consumer type/id 与有效 PID | consumer 可通过 _pid 指定 PID；普通 producer 取 plane.process_id；平台 launch 还有特殊处理。 | — |
| XSpace contexts and export | `CommonPjRtClient::CreateProfiledFuture` | `upstream/xla/xla/pjrt/common_pjrt_client.cc#L817` | Future 与 callee 名称 → 带 block start/end profiling 的 Future | 两个短 TraceMe 事件可同名，关联 ID 经 ProfilingKeys 传递；producer dur 不等于整个等待时间。 | ↓`xla.trace-context-consumer`, `xla.trace-context-producer` |
| XSpace contexts and export | `ThreadpoolEventCollector::RecordEvent` | `upstream/xla/xla/tsl/profiler/backends/cpu/threadpool_listener.cc#L56` | 调度事件 ID → ThreadpoolListener::Record 瞬时 producer | context type 为 ThreadpoolEvent；它不是任务执行区间。 | — |
| XSpace contexts and export | `ThreadpoolEventCollector::StartRegion` | `upstream/xla/xla/tsl/profiler/backends/cpu/threadpool_listener.cc#L63` | 与调度相同的 ID → StartRegion 瞬时 consumer | 本例九条关联跨线程；StopRegion 没有这个 context ID，不能混作同一终点。 | — |
| XSpace contexts and export | `TraceMeRecorder::NewActivityId` | `upstream/xla/xla/tsl/profiler/backends/cpu/traceme_recorder.cc#L259` | 线程局部和进程内计数器 → 高 32 位线程、低 32 位事件的 ID | 计数器存在复用边界；审计另以 XSpace scope 和 process/type 作为命名空间。 | ↑`xla.trace-new-activity` |
| XSpace contexts and export | `TraceMeConsumer : public TraceMe {` | `upstream/xla/third_party/tsl/tsl/profiler/lib/connected_traceme.h#L99` | 事件名、与 producer 相同的 type/id → _ct/_c metadata | 在本线程创建独立 consumer 事件；对应关系不依赖相同事件名。 | ↑`xla.profiled-future` |
| XSpace contexts and export | `TraceMeProducer : public TraceMe {` | `upstream/xla/third_party/tsl/tsl/profiler/lib/connected_traceme.h#L76` | 事件名、context type 与可选 ID → _pt/_p metadata 与 context id | 未提供 ID 时调用 NewActivityId；context type=0 是有效 Generic 值。 | ↑`xla.profiled-future` ↓`xla.trace-new-activity` |
| XSpace contexts and export | `{` | `upstream/xla/third_party/tsl/tsl/profiler/lib/context_types.h#L24` | context type 枚举 → Generic=0、ThreadpoolEvent=15 等 | 本轮 matcher 只覆盖 Generic 与 ThreadpoolEvent；TPU launch 的特殊 PID 处理未外推。 | — |
| XSpace contexts and export | `NewActivityId` | `upstream/xla/third_party/tsl/tsl/profiler/lib/traceme.h#L337` | 新的 trace activity 请求 → recorder 提供的 int64 activity id | 转发到 recorder，不是跨进程/跨文件的永久唯一标识。 | ↑`xla.trace-context-producer` ↓`xla.trace-activity-id` |
| XSpace contexts and export | `TimestampPs` | `upstream/xla/xla/tsl/profiler/utils/xplane_visitor.h#L199` | XLine timestamp_ns 与 event offset_ps → 同一坐标系的 timestamp_ps | 以整数计算 ns*1000+offset_ps，不把 line-relative offset 单独当全局时间。 | — |
| XSpace contexts and export | `DisplayId` | `upstream/xla/xla/tsl/profiler/utils/xplane_visitor.h#L332` | line display_id / id → 显示资源 ID | display_id 非零时优先，否则回退完整 line id；JSON converter 再转为 uint32。 | — |
| XSpace contexts and export | `{` | `upstream/xla/third_party/tsl/tsl/profiler/protobuf/xplane.proto#L118` | metadata_id 与 oneof value → int64/uint64/ref/string 等值 | ref_value 指向同 plane 的 stat metadata；必须保留 uint64 精度。 | — |

### `llvm` — 4 个入口

| 层次 | 符号 | 位置 | 输入 → 输出 | 约束 | 关系 |
|---|---|---|---|---|---|
| LLVM machine code | `CodeGenTargetMachineImpl::addPassesToEmitMC` | `upstream/llvm-project/llvm/lib/CodeGen/CodeGenTargetMachineImpl.cpp#L262` | legacy PM、MCContext、目标输出流 → 目标 codegen/MC emission passes | 生成对象代码的公开实现入口；机器码不是目标硬件微架构或性能的证明。 | ↑`xla.emit-object` |
| LLVM optimization pipeline | `PassBuilder::buildPerModuleDefaultPipeline` | `upstream/llvm-project/llvm/lib/Passes/PassBuilderPipelines.cpp#L1755` | OptimizationLevel、LTO phase → ModulePassManager pipeline | 具体 pass 序列取决于固定 LLVM 与参数；当前 wheel 的 .ll 前后差异不逐项证明这些固定源码 pass 已执行。 | ↑`xla.llvm-pass-manager` |
| ORC compile layer | `IRCompileLayer::emit` | `upstream/llvm-project/llvm/lib/ExecutionEngine/Orc/IRCompileLayer.cpp#L28` | MaterializationResponsibility 与 ThreadSafeModule → 提交到对象层的 compiled buffer | XLA 配置的 Compile 为 IrCompiler；失败时 failMaterialization，而非返回可用 kernel。 | ↓`xla.ir-compile` |
| ORC lookup | `ExecutionSession::lookup` | `upstream/llvm-project/llvm/lib/ExecutionEngine/Orc/Core.cpp#L1794` | Dylib 搜索顺序、符号集与 RequiredState → 等待解析完成的 SymbolMap | 公开 ORC lookup/materialization 入口，未在本轮直接绑定回调或测量加载时间。 | ↑`xla.object-lookup` |

### `xprof` — 11 个入口

| 层次 | 符号 | 位置 | 输入 → 输出 | 约束 | 关系 |
|---|---|---|---|---|---|
| Profiler CLI | `get_roofline_model` | `upstream/tooling/xprof/plugin/xprof/cli/tools/get_roofline_model_tool.py#L24` | session_id、top_n、group_by、bypass_cache → program/device/top_operations JSON 摘要 | 此固定版 del group_by；获取 roofline_model.json 后 fallback roofline_model。缺失值部分转 0，必须结合原始数据判断。 | — |
| Profiler aggregation | `GenerateRooflineModelProgramRecord` | `upstream/tooling/xprof/xprof/convert/op_stats_to_roofline_model.cc#L145` | OpMetricsDb、OpStats、record type 和总时间 → 聚合后的 program 记录 | 跳过 MayHaveInnerOps 类别避免重复计数；infeed/outfeed 由选项控制。 | ↓`xprof.roofline-record` |
| Profiler analysis | `ConvertOpStatsToRooflineModel` | `upstream/tooling/xprof/xprof/convert/op_stats_to_roofline_model.cc#L272` | Combined OpStats 与 RooflineModelOptions → profile/step 记录和 diagnostics | 输入必须已有运行环境及成本/时间数据；函数不是从单张 StableHLO 推导全部设备信息。 | ↑`xprof.roofline-processor` |
| Profiler analysis | `SetRooflineMetrics` | `upstream/tooling/xprof/xprof/convert/op_metrics_to_record.h#L180` | OpMetrics、PerfEnv、RunEnvironment 与 record → 吞吐、各级内存强度和 bottleneck | measured_flop_rate 是 flops_v2/实测时间，不自动等于硬件 counter；无分层字节时此版按 HBM 处理。 | ↑`xprof.roofline-record` |
| Profiler analysis | `ConvertOpMetricsToRooflineModelRecord` | `upstream/tooling/xprof/xprof/convert/op_stats_to_roofline_model.cc#L62` | OpMetrics、PerfEnv/RunEnvironment、record type 和总时间 → 时间、强度、资源上限与效率记录 | 利用率使用资源最大值；异步 copy 的估计可能超过 1，不应一律裁剪或解释为测量正确。 | ↑`xprof.roofline-program` ↓`xprof.roofline-metrics` |
| Profiler conversion | `RooflineModelProcessor::ProcessSession` | `upstream/tooling/xprof/xprof/convert/roofline_model_processor.cc#L39` | SessionSnapshot 和 ToolOptions → 合并 XSpace→OpStats→RooflineModel 的 JSON | 分别生成包含/排除 infeed/outfeed 的记录；没有在本轮执行 XProf native converter。 | ↓`xprof.roofline-db` |
| Profiler cost conversion | `HloCostAnalysisWrapper::GeneratePerformanceInfo` | `upstream/tooling/xprof/xprof/utils/hlo_cost_analysis_wrapper.cc#L60` | HloInstruction 与 backend cost analysis wrapper → PerformanceInfo 的 FLOPs/字节和内存分解 | 内存分解仅输出正值，跳过 0 与 -1；TPU/GPU 获取 cost 的路径需分别追踪。 | — |
| Profiler cost conversion | `ValidHloCost` | `upstream/tooling/xprof/xprof/utils/cost_utils.h#L30` | cost 数值，包括 -1 未知 sentinel → -1 映射为 0，其他值保持原样 | 下游出现 0 不足以证明该算子没有计算/流量；需要追踪原始 cost 可用性。 | — |
| XSpace contexts and export | `PreprocessSingleHostXSpace` | `upstream/tooling/xprof/xprof/convert/preprocess_single_host_xplane.cc#L34` | XSpace 与 step_grouping 等选项 → 预处理、flow 和分组后的 XSpace | 仅 SOURCE-ONLY；有 step_grouping 且未分组时调用 AddFlowsToXplane，本轮没有构建/执行此 XProf 管线。 | ↓`xla.context-add-flows` |
| XSpace contexts and export | `ConvertXStatsToTraceEventArguments` | `upstream/tooling/xprof/xprof/convert/xplane_to_trace_container.cc#L98` | XEvent stats 与 raw arguments → flow/group/is_async 等特殊字段 | 由预处理后的 kFlow 创建 flow/async event；未实际运行此 converter。 | — |
| XSpace contexts and export | `WriteEvent` | `upstream/tooling/xprof/xprof/convert/trace_viewer/trace_events_to_json.h#L317` | TraceEvent 与 flow 方向 → FlowV2 bind_id/flow_in/flow_out 或 async JSON | 与本轮派生的 legacy s/f overlay 是不同导出路径；仅 SOURCE-ONLY。 | — |

## 关系清单

| caller | → callee | 种类 | 调用位置 | 条件 |
|---|---|---|---|---|
| `xla.mlir-to-hlo` | `xla.gspmd-detection` | direct | `upstream/xla/xla/pjrt/mlir_to_hlo.cc#L110` | build options 开启 Shardy；混合旧式 IR 兼容检查。 |
| `xla.mlir-to-hlo` | `xla.sdy-export-for-gspmd` | direct | `upstream/xla/xla/pjrt/mlir_to_hlo.cc#L119` | 检测到需 GSPMD 的 IR 后先关闭 Shardy/V3。 |
| `xla.mlir-to-hlo` | `xla.sdy-roundtrip-export` | direct | `upstream/xla/xla/pjrt/mlir_to_hlo.cc#L131` | 不按 use_shardy flag 跳过，纯 StableHLO 时可无 SDY 操作可处理。 |
| `xla.cpu-spmd-gate` | `xla.shardy-pass` | pass registration | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L638` | num_partitions>1 且 use_shardy=true。 |
| `xla.cpu-spmd-gate` | `xla.shardy-pass` | pass registration | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L676` | num_partitions<=1，显式 runSdyShardingPropagation=false。 |
| `xla.shardy-pass` | `xla.shardy-propagate` | direct | `upstream/xla/xla/service/spmd/shardy/shardy_xla_pass.cc#L521` | runSdyShardingPropagation=true。 |
| `xla.shardy-pass` | `xla.shardy-replace-computations` | direct | `upstream/xla/xla/service/spmd/shardy/shardy_xla_pass.cc#L538` | 完成 StableHLO→HLO 转换后。 |
| `xla.shardy-propagate` | `xla.shardy-dump-gate` | direct | `upstream/xla/xla/service/spmd/shardy/shardy_xla_pass.cc#L330` | — |
| `xla.shardy-propagate` | `xla.sdy-roundtrip-import` | direct | `upstream/xla/xla/service/spmd/shardy/shardy_xla_pass.cc#L409` | production 分支；测试 importMhloShardings=false。 |
| `xla.shardy-propagate` | `shardy.propagation` | direct | `upstream/xla/xla/service/spmd/shardy/shardy_xla_pass.cc#L421` | 完成 import 后。 |
| `xla.shardy-propagate` | `xla.sdy-final-export` | direct | `upstream/xla/xla/service/spmd/shardy/shardy_xla_pass.cc#L430` | SDY propagation/export 后。 |
| `shardy.propagation` | `shardy.import-pipeline` | direct | `upstream/shardy/shardy/dialect/sdy/transforms/propagation/propagation_pipeline.cc#L64` | — |
| `shardy.propagation` | `shardy.export-pipeline` | direct | `upstream/shardy/shardy/dialect/sdy/transforms/propagation/propagation_pipeline.cc#L110` | 传播和函数 call 处理后。 |
| `shardy.user-priority` | `shardy.op-priority` | direct | `upstream/shardy/shardy/dialect/sdy/transforms/propagation/user_priority_propagation.cc#L238` | 首先 priority 0。 |
| `shardy.export-pipeline` | `shardy.minimal-partitioner` | direct | `upstream/shardy/shardy/dialect/sdy/transforms/export/export_pipeline.cc#L115` | avoidExportForPartitioning=false。 |
| `jax.execute-replicated` | `jaxlib.execute-sharded` | interface | `upstream/jax/jax/_src/interpreters/pxla.py#L413` | Python effect/host-callback 分支。 |
| `jax.execute-replicated` | `jaxlib.execute-sharded` | interface | `upstream/jax/jax/_src/interpreters/pxla.py#L420` | Python 无 effect 分支。 |
| `jax.pjit-fast-execute` | `ifrt.pjrt-execute` | interface | `upstream/jax/jaxlib/pjit.cc#L852` | fastpath 成立且动态 IFRT 实现是 PjRtLoadedExecutable；不经过 execute_sharded。 |
| `jaxlib.execute-sharded` | `jaxlib.execute-local` | direct | `upstream/jax/jaxlib/py_executable.cc#L512` | — |
| `jaxlib.execute-local` | `ifrt.pjrt-execute` | interface | `upstream/jax/jaxlib/py_executable.cc#L453` | 动态 IFRT 实现为 PjRtLoadedExecutable。 |
| `ifrt.pjrt-execute` | `pjrt.capi-execute` | interface | `upstream/xla/xla/python/pjrt_ifrt/pjrt_executable.cc#L1007` | 非 portable；底层 provider 为 PjRtCApiLoadedExecutable。 |
| `pjrt.capi-execute` | `pjrt.capi-execute-args` | direct | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L3450` | — |
| `pjrt.capi-execute` | `pjrt.c-execute-args` | C API | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L3472` | 进入 plugin 函数指针；不声称真实 libtpu 内部使用开源 wrapper_impl。 |
| `pjrt.capi-execute` | `pjrt.event-to-future` | direct | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L3478` | 请求了设备完成事件。 |
| `pjrt.event-to-future` | `pjrt.event-callback-contract` | C API | `upstream/xla/xla/pjrt/c/pjrt_c_api_helpers.cc#L400` | 注册 host callback，回调错误状态与注册错误分开。 |
| `jaxlib.array-wait` | `jaxlib.buffers-wait` | direct | `upstream/jax/jaxlib/py_array.cc#L868` | array 未删除。 |
| `jaxlib.buffers-wait` | `ifrt.array-ready` | interface | `upstream/jax/jaxlib/util.cc#L63` | 单数组且实现为 PjRtArray。 |
| `jaxlib.buffers-wait` | `ifrt.values-ready` | interface | `upstream/jax/jaxlib/util.cc#L71` | 多个数组且 client 为 PjRtClient。 |
| `jaxlib.buffers-wait` | `jaxlib.wait-with-signals` | direct | `upstream/jax/jaxlib/util.cc#L73` | — |
| `ifrt.array-ready` | `pjrt.buffer-ready-future` | interface | `upstream/xla/xla/python/pjrt_ifrt/pjrt_array.cc#L619` | 一个本地 buffer 且 provider 为 C API buffer。 |
| `pjrt.buffer-ready-future` | `pjrt.buffer-track-event` | direct | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L4070` | 首次创建 readiness promise。 |
| `pjrt.buffer-track-event` | `pjrt.buffer-ready-event` | direct | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L4018` | — |
| `pjrt.buffer-ready-event` | `pjrt.buffer-ready-contract` | C API | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L4000` | 首次取得 cached ready event。 |
| `pjrt.buffer-track-event` | `pjrt.event-ready-contract` | C API | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L4020` | 先查询 ready，包括失败终态。 |
| `pjrt.buffer-track-event` | `pjrt.event-callback-contract` | C API | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L4054` | 尚未 ready 时。 |
| `jax.effects-barrier` | `jax.runtime-tokens` | direct | `upstream/jax/jax/_src/api.py#L2749` | 当前线程局部 RuntimeTokenSet。 |
| `jax.block-until-ready` | `jaxlib.array-wait` | interface | `upstream/jax/jax/_src/api.py#L2764` | leaf 是本例 PyArray；其他类型按 duck typing 处理。 |
| `pjrt.capi-execute` | `pjrt.profiler-link` | direct | `upstream/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L3467` | 提交给 plugin 的 linkage producer。 |
| `pjrt.c-wrapper-execute` | `pjrt.profiler-context` | direct | `upstream/xla/xla/pjrt/c/pjrt_c_api_wrapper_impl.cc#L2325` | 仅开源 reference wrapper 的 Args extension 路径。 |
| `xla.trace-context-producer` | `xla.trace-new-activity` | direct | `upstream/xla/third_party/tsl/tsl/profiler/lib/connected_traceme.h#L86` | 调用者未提供 context_id 时。 |
| `xla.trace-new-activity` | `xla.trace-activity-id` | direct | `upstream/xla/third_party/tsl/tsl/profiler/lib/traceme.h#L339` | — |
| `xla.profiled-future` | `xla.trace-context-producer` | direct | `upstream/xla/xla/pjrt/common_pjrt_client.cc#L825` | on_block_start 回调。 |
| `xla.profiled-future` | `xla.trace-context-consumer` | direct | `upstream/xla/xla/pjrt/common_pjrt_client.cc#L834` | on_block_end 回调。 |
| `xprof.context-preprocess` | `xla.context-add-flows` | interface | `upstream/tooling/xprof/xprof/convert/preprocess_single_host_xplane.cc#L59` | step_grouping 且尚未分组；仅索引源码 API 关系，未验证匹配 XProf 构建。 |
| `xla.xplane-trace-events` | `xla.internal-trace-stat` | direct | `upstream/xla/xla/tsl/profiler/convert/xplane_to_trace_events.cc#L94` | metadata stats 和 occurrence stats 都经过此过滤。 |
| `ifrt.executable-serialize` | `xla.cpu-executable-serialize` | interface | `upstream/xla/xla/python/pjrt_ifrt/pjrt_executable.cc#L604` | 底层 executable 为 PjRtCpuExecutable 时。 |
| `xla.cpu-aot-create` | `xla.cpu-thunk-sequence-proto` | direct | `upstream/xla/xla/service/cpu/cpu_aot_compilation_result.cc#L85` | — |
| `xla.cpu-thunk-traced-execute` | `xla.cpu-thunk-trace-fields` | direct | `upstream/xla/xla/backends/cpu/runtime/thunk_executor.cc#L231` | profiler active 分支。 |
| `xla.cpu-thunk-traced-execute` | `xla.cpu-dot-execute` | interface | `upstream/xla/xla/backends/cpu/runtime/thunk_executor.cc#L234` | thunk 的动态类型为 DotThunk；当前两组梯度的 serialized kind 与 trace 对应。 |
| `xla.cpu-dot-execute` | `xla.cpu-eigen-typed` | direct | `upstream/xla/xla/backends/cpu/runtime/dot_thunk.cc#L176` | 类型分派后的 batch loop。 |
| `xla.cpu-eigen-typed` | `xla.cpu-eigen-contract` | direct | `upstream/xla/xla/backends/cpu/runtime/dot_lib.h#L102` | is_aligned=true 分支。 |
| `xla.cpu-eigen-typed` | `xla.cpu-eigen-contract` | direct | `upstream/xla/xla/backends/cpu/runtime/dot_lib.h#L107` | is_aligned=false 分支。 |
| `xla.cpu-dot-thunk` | `xla.cpu-dot-strategy` | direct | `upstream/xla/xla/service/cpu/thunk_emitter.cc#L1002` | EmitDotThunk passes allow_runtime_calls=true. |
| `xla.xplane-trace-events` | `xla.internal-trace-stat` | direct | `upstream/xla/xla/tsl/profiler/convert/xplane_to_trace_events.cc#L94` | 转换每个有值的 metadata/occurrence stat 时。 |
| `xla.ir-compile` | `xla.llvm-pass-manager` | direct | `upstream/xla/xla/backends/cpu/codegen/ir_compiler.cc#L321` | 目标机器创建成功后。 |
| `xla.ir-compile` | `xla.emit-object` | direct | `upstream/xla/xla/backends/cpu/codegen/ir_compiler.cc#L336` | LLVM IR passes 成功后。 |
| `xla.llvm-pass-manager` | `llvm.default-pipeline` | direct | `upstream/xla/xla/backends/cpu/codegen/ir_compiler.cc#L439` | 优化等级不是 O0。 |
| `xla.emit-object` | `llvm.mc-emission` | interface | `upstream/xla/xla/backends/cpu/codegen/ir_compiler.cc#L491` | 具体 TargetMachine 使用该继承实现时。 |
| `xla.jit-compile` | `xla.object-lookup` | direct | `upstream/xla/xla/backends/cpu/codegen/jit_compiler.cc#L201` | — |
| `xla.jit-compile` | `xla.compiled-function-library` | direct | `upstream/xla/xla/backends/cpu/codegen/jit_compiler.cc#L209` | lookup 成功且派发任务已结束。 |
| `xla.object-lookup` | `llvm.orc-lookup` | direct | `upstream/xla/xla/backends/cpu/codegen/object_loader.cc#L194` | — |
| `llvm.orc-ir-emit` | `xla.ir-compile` | interface | `upstream/llvm-project/llvm/lib/ExecutionEngine/Orc/IRCompileLayer.cpp#L32` | XLA 的 IRCompileLayer 配置 Compile 为 IrCompiler。 |
| `xla.kernel-thunk-call` | `xla.kernel-call-once` | direct | `upstream/xla/xla/backends/cpu/runtime/kernel_thunk.cc#L232` | call_once_ fast path。 |
| `xla.kernel-thunk-call` | `xla.kernel-launch` | direct | `upstream/xla/xla/backends/cpu/runtime/kernel_thunk.cc#L244` | 非 call_once 且没有 intra_op_threadpool。 |
| `xla.gpu-node-cost` | `xla.lhs-metadata-parser` | direct | `upstream/xla/xla/service/gpu/gpu_latency_hiding_scheduler.cc#L881` | 非 nop 且 opcode 为 custom-call。 |
| `xla.sol-node-cost` | `xla.lhs-metadata-parser` | direct | `upstream/xla/xla/service/gpu/model/sol_latency_estimator.cc#L523` | NodeCost 第一分支，所有 opcode。 |
| `xla.gpu-lhs-pipeline` | `xla.gpu-estimator-select` | direct | `upstream/xla/xla/service/gpu/gpu_hlo_schedule.cc#L708` | GPU LHS pipeline 被启用时。 |
| `xla.gpu-lhs-pipeline` | `xla.lhs-pass` | pipeline | `upstream/xla/xla/service/gpu/gpu_hlo_schedule.cc#L833` | 选定 estimator 已放入 SchedulingContext。 |
| `jax.overlap-schedule` | `jax.overlap-control` | direct | `upstream/jax/jax/experimental/overlap.py#L29` | 相邻值逐对调用。 |
| `xla.cpu-async-pipeline` | `xla.async-custom-rewriter` | pipeline | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L610` | use_legacy_collectives=false。 |
| `xla.cpu-async-pipeline` | `xla.async-to-sync` | pipeline | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L613` | Config(HloPredicateTrue)。 |
| `jax.matmul` | `jax.dot-general` | direct | `upstream/jax/jax/_src/numpy/tensor_contractions.py#L273` | — |
| `jax.dot-general` | `jax.dot` | direct | `upstream/jax/jax/_src/lax/lax.py#L2540` | — |
| `jax.pjit-lower` | `jax.lower-sharding` | direct | `upstream/jax/jax/_src/pjit.py#L1240` | — |
| `jax.cached-mlir-lowering` | `jax.mlir-module` | direct | `upstream/jax/jax/_src/interpreters/pxla.py#L756` | — |
| `jax.backend-compile` | `jaxlib.compile` | binding | `upstream/jax/jax/_src/compiler.py#L395` | 普通 backend 路径；CompileOnlyPyClient 分支单独处理。 |
| `jaxlib.compile` | `jaxlib.ifrt-call` | direct | `upstream/jax/jaxlib/py_client.cc#L495` | — |
| `jaxlib.ifrt-call` | `ifrt.compile` | interface | `upstream/jax/jaxlib/py_client.cc#L412` | 选用 PjRtCompiler 实现时；IFRT 也有其他 compiler。 |
| `ifrt.compile` | `ifrt.pjrt-create` | direct | `upstream/xla/xla/python/pjrt_ifrt/pjrt_compiler.cc#L108` | — |
| `ifrt.pjrt-create` | `pjrt.c-api-compile` | interface | `upstream/xla/xla/python/pjrt_ifrt/pjrt_executable.cc#L769` | PJRT client 为 C API adapter 时；CPU client 可有其他实现。 |
| `pallas.dispatch` | `pallas.tpu-lower` | direct | `upstream/jax/jax/_src/pallas/pallas_call.py#L903` | TPU backend，非 interpret 默认分支。 |
| `pallas.tpu-lower` | `mosaic.module` | direct | `upstream/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L435` | — |
| `xla.cpu-codegen` | `xla.cpu-schedule` | direct | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L1832` | — |
| `xla.cpu-codegen` | `xla.cpu-buffers` | direct | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L1845` | — |
| `jaxlib.cost-binding` | `xla.cpu-cost-factory` | interface | `upstream/jax/jaxlib/xla_compiler.cc#L93` | 当传入 client 为 PjRtCpuClient 时。 |
| `xprof.roofline-processor` | `xprof.roofline-db` | direct | `upstream/tooling/xprof/xprof/convert/roofline_model_processor.cc#L45` | — |
| `xprof.roofline-program` | `xprof.roofline-record` | direct | `upstream/tooling/xprof/xprof/convert/op_stats_to_roofline_model.cc#L178` | — |
| `xprof.roofline-record` | `xprof.roofline-metrics` | direct | `upstream/tooling/xprof/xprof/convert/op_stats_to_roofline_model.cc#L93` | — |
| `jax.mlir-module` | `jax.donation-aliases` | direct | `upstream/jax/jax/_src/interpreters/mlir.py#L1384` | 满足该 lowering 的 donation/alias 分支条件时。 |
| `xla.cpu-fusion-decision` | `xla.common-fusion-decision` | direct | `upstream/xla/xla/service/cpu/cpu_instruction_fusion.cc#L473` | 前面的 CPU 专用 early return 未结束判定时。 |
| `xla.cpu-buffers` | `xla.buffer-assignment-entry` | direct | `upstream/xla/xla/service/cpu/cpu_compiler.cc#L2464` | — |
| `xla.buffer-reuse` | `xla.buffer-interference` | direct | `upstream/xla/xla/service/buffer_assignment.cc#L2029` | total_order_scheduled 分支；另有 partial ordering fallback。 |

