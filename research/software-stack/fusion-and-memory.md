# Fusion、donation 与 buffer 复用

对应 R02/R05/R06。脚本 [fusion_memory_probe.py](fusion_memory_probe.py) 的
`fusion-memory-002` 包含 11 组 CPU 对照和 2028 个产物，保存实际 HLO pass dump、LLVM IR、
allocation/live-range 报告、数值与运行时指针观察。
[verify_fusion_memory.py](verify_fusion_memory.py) 独立复查产物、图、分配、存活区间与参考输出；
结果见 [fusion-memory-results.json](fusion-memory-results.json)。

本节表格保留历史 `RUN-CPU + VERSION-SKEW` 来源。新增 [源码 003 基线](source-runtime-baseline.md)
已对 11 组样本完成独立复验，2,043 个新产物绑定匹配 native 字节，结论及 logical peak 差异仍在。
旧 capture 没有被改写。下面的教学程序没有替代 kickoff 要求的真实推理 fusion/split 或 TPU 验收。

## Fusion 与存储复用是不同优化

向量 `x[1024]` 的程序是 `tanh(cos(sin(x)))`。三个版本均通过 NumPy float64 参考。

| 编译选项 | 最终 entry HLO | 编译器 temp bytes | 模型 bytes accessed |
|---|---|---:|---:|
| 默认 | 1 个 fusion，内部包含 sin/cos/tanh | 0 | 8192 |
| 禁用 `fusion` | 3 个单算子 fusion wrapper | 0 | 24576 |
| 禁用 `fusion,fusion-wrapper` | 独立 sine/cosine/tanh | 0 | 24576 |

这些 bytes 是编译器模型值，不是实测 DRAM 流量；本实验也没有测量三者的性能比。
只看到 HLO 中有 `fusion` 不足以判断多个算子是否融合：
[FusionWrapper](../../upstream/xla/xla/service/cpu/fusion_wrapper.cc) 会为后端 emitter 包装单个算子。
实际 pass dump 分别展示了 `fusion` 与 `fusion-wrapper` 前后的结构变化，路径保存在结果 JSON。

未融合版本的 allocation 0 只有 4096 bytes，却先后存放三个值，offset 均为 0：

| 值 | 逻辑存活区间 | allocation / offset |
|---|---|---|
| sin.1 | 1–2 | 0 / 0 |
| cos.1 | 2–3 | 0 / 0 |
| tanh.1 | 3–4 | 0 / 0 |

input 另在 allocation 1，占 4096 bytes。相邻算子在合法的最后一次读取/写入边界复用输出
存储，因此不需要额外 temp allocation；这不等于没有中间值或没有中间读写。
逻辑时钟是 schedule 中的指令序号，不是时间戳或微秒。

## 给定 HLO，如何判断 fusion

首先确定它属于哪个 backend、哪个编译阶段、启用了哪些 pass/emitter/library，然后检查
producer/consumer、使用次数、shape/layout、别名及效果约束，最后用真实 before/after
验证该候选是否仍存在并被融合。数学上能组合并不等于编译器会融合。

固定 CPU 的 [CpuInstructionFusion::ShouldFuse](../../upstream/xla/xla/service/cpu/cpu_instruction_fusion.cc)
提供了具体入口：

- 跳过部分 custom fusion/call 内部；拒绝普通非标量 constant producer。
- 限制可用 elemental emitter 表达的 producer/consumer；库 GEMM 与普通循环 fusion 是不同路径。
- 对 expensive producer、元素重复使用和代码复制成本做判断；还限制某些 concatenate 和 reduction 组合。
- 调用 [公共 InstructionFusion 判断](../../upstream/xla/xla/service/instruction_fusion.cc)，检查 root
  边界、复制成本和 in-place 合法性等条件。
- 特定非复数、无 batch 的向量输出 dot+add 有单独条件；不能把某条注释概括为所有 dot 的规则。

CPU pipeline 在普通 instruction fusion 前还会做库重写，具体组装见
[cpu_compiler.cc](../../upstream/xla/xla/service/cpu/cpu_compiler.cc)。本次小矩阵程序
`max(A[4,8]@W[8,6]+B[4,6],0)` 的 dot 已变成 `kind=kCustom` 的 `__ynn_fusion`，
后面的 add/max 另形成 loop fusion。禁用 `fusion,fusion-wrapper` 后，库 fusion 仍在，
且出现一个 96-byte broadcast 临时量：编译器 accounted bytes 从 512 增至 608。
因此上述两个开关没有“关闭一切 fusion”，也不能直接移植为 TPU fusion 控制结论。

## Donation 的四个观察层次

1. `donate_argnums` 表达调用方允许消耗输入的意图。
2. JAX lowering 生成 donor/alias 信息，backend 形成最终 HLO alias 与 buffer assignment。
3. 运行时根据实际所有权、外部引用等条件执行或调整复用。
4. 通过 input invalidation、pointer equality 和仍可见的数组存储核对具体运行。

向量数据由一个 CPU JIT producer 生成，避免默认输入构造方式干扰所有权观察。
除“保留 view”对照外，不提前把输入转为 NumPy view。运行后记录指针，地址仅保存在忽略的
原始 capture；公开 JSON 只保留相等关系。

| 情形 | alias bytes | temp bytes | 静态 accounted bytes | 执行后可见 payload bytes | 原指针复用 / 输入失效 |
|---|---:|---:|---:|---:|---|
| chain，无 donation | 0 | 0 | 8192 | 8192 | 否 / 否 |
| chain，donation | 4096 | 0 | 4096 | 4096 | 是 / 是 |
| chain，donation 且保留 NumPy view | 4096 | 0 | 4096 | 8192 | 否 / 否 |
| reshape，无 donation | 0 | 0 | 8192 | 8192 | 否 / 否 |
| reshape，donation | 4096 | 4096 | 8192 | 4096 | 是 / 是 |
| reduction，donation 未使用 | 0 | 4224 | 8324 | 4100 | 否 / 否 |

静态 accounted bytes 使用 `arguments + outputs - aliases + temporaries`，并与 allocation
表中排除 constant/thread-local 后的总量核对。执行后 payload 只统计仍可见输入与输出的
不同指针及大小；它不是执行过程峰值，也没有包含所有 allocator、代码或 runtime 开销。

保留 NumPy view 时，编译计划完全相同，但本次运行没有复用原存储，view 内容仍保持原值。
固定公共 PJRT 的 [donation hold](../../upstream/xla/xla/pjrt/abstract_tracked_device_buffer.cc)
也存在外部引用检查；当前 wheel 的具体 fallback 调用路径仍需匹配源码 binary 验证。
本次输入仍有效不是可依赖的跨版本 API 保证，调用方仍应遵守 donation 后不再使用输入的合同。
公开 [JAX donation 说明](https://docs.jax.dev/en/latest/buffer_donation.html) 可作接口背景。

reshape 程序是 `(x+1).reshape(32,32)`。donation 版本用了 4096-byte 中间量，再 copy 到
与输入共享的输出 allocation；所以指针复用成功，静态 accounted bytes 仍与无 donation
相同。不能只从输出/输入 shape 相等判断可否 donation，也不能只减去 alias bytes 而忽略
新增临时量。固定 [_set_up_aliases](../../upstream/jax/jax/_src/interpreters/mlir.py) 除精确匹配外
还有相同元素数量的 fallback，最后仍由 backend 验证实际可用性。

reduction 是 `sum(x*x)`，输出为一个 scalar。本次没有合适的输出接受整个输入，出现
`Some donated buffers were not usable`，alias bytes 为 0。identity 对照也保存了显式输出
copy 的结果：程序值相同不意味着公共输入/输出一定共享物理存储。

## 从 allocation 追到源码

[CPU CreateHloSchedule/CreateBufferAssignment](../../upstream/xla/xla/service/cpu/cpu_compiler.cc)
先选 schedule，再以 SequentialHloOrdering、AliasInfo、buffer sizes 与 alignment 建立分配。
[HloLiveRange](../../upstream/xla/xla/hlo/utils/hlo_live_range.cc) 计算逻辑区间。
[BufferAssigner](../../upstream/xla/xla/service/buffer_assignment.cc) 的候选复用检查还包括空间/颜色
兼容、容量、只读、live-out 和可复用标志；区间端点相接时再检查 producer/user 是否能共享。
copy 等操作有额外限制，不能仅凭两个区间“看起来不重叠”就强行原地化。

还要分开“同一 allocation”和“同一片字节”。在已有 `cpu-matmul-003` 的 batch 梯度中，
两个 288-byte temporary 分别位于 allocation 5 的 offset 0、320，它们的存活区间重叠。
该 608-byte allocation 是两片存储加 32-byte 间隙；它不是两个 temporary 原地共享同一片空间。
另一个 384-byte 输出 allocation 则先容纳 288-byte forward 结果，再容纳后来的 gradient，
其区间分开，属于真正的时序复用。原始文件的指纹仍由 CPU capture validator 校验。

## 诊断中的 peak 标签也要核对

本次 reduction 的 `live-range.txt` 把逻辑时刻 4 标为 peak，列出的值共 4232 bytes；
但按同一文件的闭区间逐时刻相加，时刻 3 为 8324 bytes：

| 逻辑时刻 | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| 打印区间对应的逻辑值总量 | 4096 | 4100 | 8196 | 8324 | 4232 | 4100 |

独立验证器会记录这个差异，而不是把标签当成验收结论。固定源码
`ComputePeakMemoryMoment` 按事件逐个更新总量，同一时刻先处理 start，再处理 `end+1`；
这可能把瞬时过渡状态选为 peak，而随后 `ToString` 展示的闭区间集合不同。
这是捕获重算与源码检查得到的诊断问题线索，尚未在匹配源码 binary 上修复/验收。

即使选点修正，逻辑值总量也不能直接替代物理内存峰值：未融合 chain 的相邻值在边界时
共同出现，但分配表允许它们复用字节。最终应联合 allocation/offset、alias、存活区间、
runtime 所有权以及目标设备测量来解释峰值，不能把任一单项报告当作完整答案。

```bash
.venv/bin/python -B research/software-stack/verify_fusion_memory.py --selftest
.venv/bin/python -B research/software-stack/fusion_memory_probe.py --output artifacts/jax-stack/fusion-memory-replay
```

复现使用新目录。下一步是在指定业务上选择候选节点，应用合法 fusion/split 改写，
取得精度和真实内存/设备对照；这依赖仍待回答的 U01–U03。
