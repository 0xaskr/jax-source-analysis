# JAX 软件栈：开放探索

这里按主题探索整个 JAX 软件栈，不以某个固定调用或 LLO 产物为目标。原 `research/jax-stack/` 的材料按阅读主题和代码依赖重新归档；旧实验的版本与证据范围不因搬迁而改变。

| 主题目录 | 从哪里读起 |
|---|---|
| [jax-repo](jax-repo/README.md) | JAX 仓库布局、Python 与 jaxlib/XLA 源码归属 |
| [source](source/overview.md) | 组件全貌、源码索引、构建输入与运行环境 |
| [compiler](compiler/shardy-round-trip.md) | Shardy、HLO/pass、属性和 CPU LLVM/对象文件 |
| [runtime](runtime/tpu-runtime-boundary.md) | PJRT/IFRT、公开 TPU 边界、CPU executable |
| [profiling](profiling/profiling.md) | host/编译/设备事件、XSpace 与 Roofline |
| [performance](performance/fusion-and-memory.md) | Fusion、内存、调度与 latency 模型 |
| [pallas](pallas/pallas-comparison.md) | Pallas 的通用对照；[示例](examples/pallas_add_stablehlo.py)用于观察两层表示 |
| [workload](workload/sglang-v7-workload.md) | 既有 SGLang v7 工作负载材料 |
| [figures](figures/overview-software-stack.svg) | 栈、编译控制链与 Pallas 表示图 |
| [tools](tools/README.md) | 共用探针、验证器、补丁和历史 JSON 结果 |
| [history](history/README.md) | 旧计划、状态和覆盖复查；仅供追溯 |

共用的 `matmul_probe.py` 留在 [tools](tools/matmul_probe.py)，因为多个探针导入它的 capture/指纹函数。针对固定 `jnp.matmul` 与双缓冲 Pallas kernel 的纵向分析在 [call-to-llo](../call-to-llo/README.md)。引用旧结果时分别标明源码检查、CPU 执行、TPU 模拟、离线 TPU 编译和真实 TPU 执行；Mosaic TPU MLIR 不等于 LLO。
