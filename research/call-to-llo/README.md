# 固定调用到 LLO

这里按**调用**归档，而不是按编译器组件归档。两条路径分别有自己的入口，后续证据应能关联到同一次编译、固定输入、匹配的源码/运行时和目标 TPU。当前都尚未取得目标 LLO。

| 调用目录 | 已保存的阶段 | 尚缺的阶段 |
|---|---|---|
| [`matmul/`](matmul/README.md) | `jnp.matmul` → Jaxpr/`dot_general_p` → StableHLO 的源码走读，以及独立的 CPU pass/执行材料 | 与该调用对应的目标 TPU 后端 dump 和 LLO |
| [`pallas/`](pallas/README.md) | `manual_double_buffer_add` 的源码、外层 `tpu_custom_call` StableHLO 与 payload 内 Mosaic TPU MLIR；CPU host lowering/模拟记录 | 匹配目标 TPU 的编译、LLO、真机执行与设备观测 |

通用的编译和运行机制见 [软件栈开放探索](../software-stack/index.md)。CPU 执行、TPU 模拟、离线 TPU 编译与真实 TPU 执行分别取证；**Mosaic TPU MLIR 不是 LLO**。若运行时二进制与引用源码不匹配，记录 `VERSION-SKEW`。
