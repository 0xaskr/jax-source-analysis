# StableHLO/HLO 属性、编辑与 cost model

对应 R09、R11，并为 R10 提供可用输入。脚本
[attributes_cost_probe.py](attributes_cost_probe.py) 的成功捕获为 `attributes-cost-002`，
60 个产物经 [verify_attributes.py](verify_attributes.py) 独立复查；结果见
[attributes-results.json](attributes-results.json)。本节运行证据为 `RUN-CPU + VERSION-SKEW`。

结论是：**现有接口支持“携带自定义属性，同时运行默认 cost analysis”；自定义属性不会自动
定义新的 cost 语义。** 还要区分 JAX metadata、任意 MLIR 属性和 native HLO 属性。

## 六组 metadata 对照

固定输入 `A[4,8] @ W[8,6]`，FP32。标签为 `research_tag="matmul"`、
`research_flops=999999`、`research_boolean=True`。

| 写法 | StableHLO 属性所在位置 | 导出 HLO 的带标签指令数 | 默认 FLOPs |
|---|---|---:|---:|
| 无标签 | 无 | 0 | 384 |
| `set_xla_metadata` context | dot_general | 1 | 384 |
| `set_xla_metadata(result, ...)` | 结果的 producer dot_general | 1 | 384 |
| `xla_metadata_call2` | func.call | 1 | 384 |
| call 的梯度，`ad_metadata="same"` | 前向及反向调用 | 2 | 792 |
| call 的梯度，`ad_metadata="drop"` | 前向调用 | 1 | 792 |

梯度对象是 `d sum((A@W)^2) / dA`，参考式为 `2*(A@W)@W.T`。
这里没有对 W 求导，不能与另一实验的双参数梯度 FLOPs 混用。
六组结果均通过 NumPy float64 参考，容差 `rtol=atol=2e-5`。

四组前向的 lowering/compiled cost dict 相同，FLOPs 为 `2*4*8*6=384`，
默认字节估计为 `(4*8+8*6+4*6)*4=416`。标签数值 `999999` 仍在 HLO，
但没有变成 FLOPs。梯度默认成本 792 是该图的分析结果，不是标签指定值。

生产入口是 [set_xla_metadata / xla_metadata_call2](../../upstream/jax/jax/_src/xla_metadata.py)：
Python int/float/string 转成字符串，bool 转成小写 `true/false`；call 版本在 lowering 时
给 `func.call` 附加 `mhlo.frontend_attributes`。编译后内联和 fusion 可把属性传到其他
指令，不能要求优化前后标签节点数不变。本例前向优化后的两处标签不代表两次 matmul。

## 直接编辑 MLIR 与使用 API 的区别

脚本解析一份独立 StableHLO module，保留原 Lowered 对象与缓存，然后在 dot 上加入：

```mlir
research.raw = "raw-visible"
mhlo.frontend_attributes = {
  research_string = "string-visible",
  research_bool = true,
  research_integer = 123 : i64
}
```

Module verifier 通过；导出 HLO 后字符串、布尔值保留，`research_integer` 与
`research.raw` 未保留。固定 XLA 的
[CreateFrontendAttributes](../../upstream/xla/xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc)
显式处理 StringAttr/BoolAttr。因此 JAX API 传入 Python 整数会保留其字符串表示，
与直接写 MLIR IntegerAttr 的结果不同。这个结论限于所测属性和转换路径，
不是“所有非字符串属性都不能存在于 StableHLO”。

另一个实验使用 native HLO wrapper 的 `get_frontend_attribute` /
`set_frontend_attribute` 修改字符串标签，序列化、重新解析后仍可读出。
绑定实现见 [hlo.cc](../../upstream/xla/xla/python/hlo.cc)，接口声明见
[_hlo.pyi](../../upstream/xla/xla/python/_hlo.pyi)。修改成功仍不代表 cost model 会消费该键。

## StableHLO 到 HLO 不只是换一种文本

| 信息 | 本次 StableHLO | 导出或优化后的 HLO |
|---|---|---|
| 表示结构 | MLIR module/func、SSA 值、方言 operation | HloModule/computation/instruction；优化后可能出现 fusion computation |
| matmul 语义 | dot dimension numbers、precision、tensor 类型 | dot 的收缩维度、operand_precision、shape；本例语义保留 |
| layout | Tensor 类型没有本例 HLO 中的显式 minor-to-major 标记 | 导出文本带 `{1,0}`，但这不等于目标 TPU 最终物理布局 |
| 名称与位置 | MLIR locations、Python source/name stack | 一部分转为 OpMetadata；导出选项和优化可能改变展示 |
| 自定义信息 | 可有方言/命名空间属性、frontend dictionary | 只保留转换器有映射或明确转发的字段 |
| backend 信息 | 前端 lowering 表示 | 优化后可有 schedule、fusion、backend config 等；哪些出现取决于阶段和 backend |

`compiler_ir("hlo")` 是导出观察点，不能冒充真实 backend pass pipeline。
本实验另保存编译后的 HLO；逐 pass 证据仍以第一轮 `cpu-matmul-003` 的 native dumps 为准。
不是每个 HLO 阶段都已有 schedule、buffer assignment 或物理布局。

## 实际执行一次 HLO 语义改写

程序先是 `A@W+B`，`B[4,6]`。脚本只把 HLO 的一个 `add` opcode 改为 `subtract`，
通过 `_hlo.hlo_module_from_text` 重新解析，序列化为 HloModuleProto，导回 MLIR，
再使用 CPU `compile_and_load` 构建 executable 并实际调用。

改写前后分别符合 `A@W+B` 和 `A@W-B`，最大绝对误差约 `3.68e-8`、`7.34e-8`；
两份输出的差等于 `2*B`。独立验证器重新解析 opcode，确认是一次 add→subtract 的变化。
另将 opcode 改成不存在的名称，HLO parser 按预期拒绝。

这证明当前私有绑定可支持此类小图编辑与执行；不证明任意添加、复制、替换指令都安全，
也不是稳定公共 JAX 图编辑 API。真实 pass 还必须保持 shape/type、effects、alias/donation、
sharding/layout、控制依赖和 metadata 合同。可逆方法是保留原 module 或重新执行原 producer，
没有修改 pinned 源码文件。

## 默认成本来自哪里，如何扩展

1. [jaxlib cost binding](../../upstream/jax/jaxlib/xla_compiler.cc) 接收 client/HloModule，
   调用该 PJRT client 的 `GetHloCostAnalysis()`，遍历 entry computation。
2. [CPU factory](../../upstream/xla/xla/pjrt/cpu/cpu_client.cc) 创建通用 HloCostAnalysis，
   指定 CPU shape byte size；没有在此配置目标吞吐 rates。
3. [HloCostAnalysis](../../upstream/xla/xla/service/hlo_cost_analysis.cc) 的 Preprocess 默认
   统计输入/输出字节，HandleDot 按结果元素数和 reduction width 算 FLOPs；Postprocess
   在配置 rates 时按最大资源时间估算 bottleneck time。

如果“可并行维度”只表示允许切分的提示，它没有改变总工作量；默认总 FLOPs 不变是合理结果。
如果希望据该属性计算**每分块**成本，则必须先定义维度、分块数量、padding、重复读、通信和
合并成本的合同，再在合适阶段消费该属性，不能把 annotation 数字直接当成硬件测量。

需要新增语义时，修改点分三层：JAX/MLIR 标记与传递规则；XLA 或具体 backend 的 cost
handler/subclass；若要进入 profiler，再修改对应 PerformanceInfo/OpMetrics 转换。
单纯在 StableHLO dictionary 添一个字段不会自动贯通这三层。相关改动应分别验证标签
存活、数值不变、分块形状/算量/字节口径，以及优化后属性的归属。

## 未知 custom-call 成本

将上一轮 Mosaic 外层 `tpu_custom_call` 交给 **CPU reference analyzer**，得到 FLOPs、
bytes、optimal_seconds 均为 `-1`。固定通用 HandleCustomCall 用它表示未知成本，
内部无运行成本的 call marker 是例外。负值不能当成零工作，也不能相除得到算术强度。
这不是对 libtpu cost model 的执行验证；TPU compiler/runtime 的路径仍需独立取证。

Roofline 已有模块与所缺输入见 [roofline.md](roofline.md)。本节尚未验证 TPU 上属性保留、
可并行维度驱动的真实分块，或自建 native binary 的行为。

```bash
.venv/bin/python -B research/jax-stack/verify_attributes.py --selftest
.venv/bin/python -B research/jax-stack/attributes_cost_probe.py --output artifacts/jax-stack/attributes-cost-replay
```
