# JAX 软件栈开放探索

这里从组件和问题出发探索 JAX 软件栈，不预设算子、kernel 或性能目标。当前目录保存的是图示入口；原有专题文档、实验结果和计划已移除。源码以本地 [`upstream/`](../../upstream) 检出为准，修订版本与来源见 [`upstream-sources.lock`](../../upstream-sources.lock)、[`source-archives.lock`](../../source-archives.lock) 和 [`env/environment.lock.json`](../../env/environment.lock.json)。

`upstream/jax` 当前位于以 JAX v0.11.1 为基础的源码注释提交。XLA、StableHLO、Shardy 和 LLVM 的本地检出采用稀疏范围；目录中找不到某个文件时，应先确认它是否尚未落地。

## 从源码进入

| 方向 | 本地入口 | 可先看的图 |
|---|---|---|
| JAX Python、tracing、Jaxpr 与 jaxlib | [JAX](../../upstream/jax) | [tracing 内部](overview/jax-tracing-internals.svg)、[组件关系](overview/jax-stack-components.svg) |
| 程序表示与分片 | [StableHLO](../../upstream/stablehlo)、[Shardy](../../upstream/shardy) | [软件栈全貌](overview/overview-software-stack.svg) |
| 编译控制与后端路径 | [XLA](../../upstream/xla) | [编译控制链](overview/overview-compile-control.svg)、[后端分叉](overview/overview-backend-paths.svg) |
| LLVM/MLIR 与 CPU 代码生成 | [LLVM](../../upstream/llvm-project) | [组件关系](overview/jax-stack-components.svg) |
| Pallas 的通用 lowering 机制 | [JAX Pallas](../../upstream/jax/jax/_src/pallas)、[XLA Mosaic](../../upstream/xla/xla/mosaic) | [Pallas 表示层次](overview/overview-pallas-lowering.svg) |

这些图帮助定位源码入口。图中的路径和行号属于制作时的记录，引用前需与当前锁定源码重新核对。源码检查、CPU 执行、TPU 模拟、离线 TPU 编译与真实 TPU 执行分别取证；运行时与所引源码不一致时标记 `VERSION-SKEW`。Mosaic TPU MLIR 不是 LLO。

固定 `jnp.matmul` 与双缓冲 Pallas kernel 的逐调用追踪见 [call-to-llo](../call-to-llo/README.md)。
