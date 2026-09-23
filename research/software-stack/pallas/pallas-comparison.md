# 普通 JAX 与两种 Pallas 解释路径

沿用 CPU matmul 的完全相同输入：`A[4,8] @ W[8,6]`，批量输入 `X[3,4,8]`，共享 `W`。
结果见 [extension-results.json](../tools/extension-results.json) 的 `pallas`；脚本为
[pallas_probe.py](../tools/pallas_probe.py)。

| 路径 | 开发者表达与控制 | StableHLO 观察 | 证据 |
|---|---|---|---|
| 普通 JAX | 数组 matmul；编译器决定实现 | 1 个 dot_general mention | RUN-CPU |
| Pallas `interpret=True` | Ref kernel、输出 tile `[2,2]`、grid `[2,3]`、输入块映射 | while 与 dynamic slice/update，1 个 dot_general mention | RUN-CPU，通用解释器 |
| Pallas `pltpu.InterpretParams` | 同一 kernel，TPU 专用语义解释；一个模拟 core | 更多 while，26 个 CPU callback custom call，仍保留 dot | SIM-TPU，运行后端是 CPU |
| `vmap` Pallas generic | 共享 W，对三份输入做 batching | 解释循环和批量 dot | RUN-CPU |

operation mentions 是词法扫描结果，不是完整语义 IR 节点计数。TPU interpreter 的 callback
管理共享内存模型、同步等语义；普通数值 primitive 仍可通过 `prim.bind` 留在 CPU 图里。
不能说这份 matmul 完全在 Python callback 内计算。

三条单矩阵路径的最大绝对误差都是约 `5.10e-8`；generic vmap 约 `2.92e-8`。
独立验证器复查 NumPy float64 参考、输出文件和 grid recorder。模拟访问覆盖六个不同坐标，
均属于模拟 core 0；访问顺序不能外推硬件并行调度。

固定源码入口见 [pallas_call 分派](../../../upstream/jax/jax/_src/pallas/pallas_call.py)、
[generic interpreter](../../../upstream/jax/jax/_src/pallas/hlo_interpreter.py)、
[TPU interpret 参数](../../../upstream/jax/jax/_src/pallas/mosaic/interpret/params.py) 与
[TPU interpreter](../../../upstream/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py)。
Typed `InterpretParams` 与布尔 `True` 进入不同分支，不能统称同一种 TPU 模拟。

`interpret=False` 在 CPU lowering 按预期抛出 `Only interpret mode is supported on CPU backend`。
`[2,2]` 教学 tile 只用于解释语义，没有声称它被 TPU 编译器或硬件接受。
较大 tile 的生产 Mosaic lowering 见 [profiling.md](../profiling/profiling.md)；它与本节小形状实验分别记录，
也仍未进行 libtpu 编译或真实 TPU 执行。

首次 `pallas-matmul-001` 失败来自探针自身：grid recorder 把 token 错当成 device ID，且没有
返回 token。修正为 `(token, coordinates, core_id) -> token` 后产生成功的 `002`。
失败日志和生产脚本保留在原目录，没有把它归类为 JAX 缺陷。

本节未验证 DMA、竞争检测、通信、硬件时钟或性能。R03 还需要匹配 TPU backend 的编译、
Mosaic/外层 HLO/LLO 分层材料、真机数值与设备 trace。
