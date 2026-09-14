# JAX 软件栈研究：lowering、Pallas 与 profiling

本轮依据 [Outline kickoff revision 51](https://outline.infiscale-tech.com/doc/research-plan-jax-kickoff-ib5QULKSS4)
新建研究。完整要求和未完成项保留在 [PLAN.md](PLAN.md)，没有沿用已删除的旧进度。

- [软件栈与 API 总览](overview.md)：组件、控制链、普通 CPU/TPU 与 Pallas 两层表示。
- [源码索引](source-index.json)：120 个入口、38 条带调用位置及分支条件的关系，含锁定的 XProf tooling 源码。
- [Pallas 对照](pallas-comparison.md)：相同输入的普通 JAX、generic 与 TPU interpret。
- [三类 profiling 区间](profiling.md)：host、编译 pass、设备标记的实验证据与验收边界。
- [扩展实验结果](extension-results.json)：四组 capture、62 个产物的独立复查。
- [Pallas/profiling Notebook](pallas-profiling.ipynb)：实算对照、trace 统计与 Mosaic 标记检查。
- [源码构建](source-build.md)：固定镜像、隔离克隆与实际构建状态。
- [LLVM、对象与 ORC](llvm-and-objects.md)：三个 ELF 对象与 HLO/LLVM 函数、序列化字节的对应。
- [自定义 pass 事件补丁](pass-event-patch.md)：规范补丁已验证应用和回滚，尚未编译加载。
- [Pass 事件验收](pass-event-acceptance.md)：cold/warm/filter 负对照、构建状态拒绝与 wheel/native 身份核对。
- [构建依赖审计](build-dependency-results.json)：三份归档、43 个补丁目标与已缓存 payload 的校验边界。
- [通信与调度](overlap-and-scheduling.md)：CPU async→sync、控制依赖和四个实验性 API 失败。
- [调度 Notebook](overlap-scheduling.ipynb) 与 [结果](overlap-results.json)：双 CPU 数值、IR、593 个产物的独立复查。
- [属性、HLO 编辑与成本](attributes-and-cost.md)：六种 metadata 情形、属性丢失边界、HLO 改写执行与成本读取。
- [属性/cost Notebook](attributes-cost.ipynb) 与 [验证结果](attributes-results.json)：原始 60 个产物复查，现有 6 个代码单元已执行。
- [latency_metadata 与模型](latency-model.md)：解析、消费者、PGLE 优先级、调度 gate；[8 组 CPU 对照](latency-metadata-results.json) 分开验证传递与成本。
- [Roofline 模块与边界](roofline.md)：XProf 处理链、硬件/时间输入、Unknown 与零值，以及修正位置。
- [Fusion 与内存复用](fusion-and-memory.md)：11 组 CPU 对照，区分 fusion、donation、实际指针、allocation 与逻辑 peak。
- [Fusion/memory Notebook](fusion-memory.ipynb) 与 [结果](fusion-memory-results.json)：2028 个产物的独立复查。
- [CPU 实验结果](cpu-results.json)：数值、pass 边界、静态 cost/memory、ELF 目标文件及 capture 指纹。
- [生产脚本](matmul_probe.py)：生成四组样本、原始 IR/日志、二进制和来源记录。
- [交互 Notebook](matmul-lowering.ipynb)：八个代码单元已在真实 Jupyter 内核中执行通过，含对象与补丁审计。
- [当前研究状态](status.json)：逐项记录 R01–R12 的证据、缺口和下一步。

## 实验与数值对照

固定随机种子 `20260914`、输入 `float32`、dot precision `HIGHEST`。
单样本使用 `A[4,8] @ W[8,6]`；batch 输入为 `X[3,4,8]`，共享同一 `W`。
参考计算在 NumPy float64 上进行；梯度参考使用解析公式，容差 `rtol=atol=2e-5`。

| 示例 | 运算与输出 | 验证 |
|---|---|---|
| matmul | `A @ W → Y[4,6]` | 与 NumPy matmul 比较 |
| vmap | `vmap(matmul, in_axes=(0,None))(X,W) → Z[3,4,6]` | 与 NumPy broadcast matmul 比较 |
| grad | `grad(sum((A@W)^2), argnums=(0,1))` | `dA=2YWᵀ`，`dW=2AᵀY` |
| jit/grad/vmap | 对 batch loss 求梯度后 JIT 编译 | `dX=2ZWᵀ`，`dW=2∑ᵦ XᵦᵀZᵦ` |

四组样本全部通过；六个结果数组的最大绝对误差约 `1.43e-7`。另外对一个独立的
`cache_matmul` 连续执行相同 shape、相同 shape、新 shape 三次调用：Python tracing
计数为 `[1,1,2]`；日志中另有两条该函数的 XLA 编译完成事件。这里把 tracing 与编译
分别取证，没有从 tracing 计数直接推导编译次数。

四组 executable 通过 `jax.experimental.serialize_executable` 序列化，在生产进程中
重新加载并执行，结果逐位一致。`.bin` 是包含 Python 包装元数据的序列化包，不能
当成纯 ELF；另外捕获的三个 `.o` 已用 `readelf` 确认是可重定位 ELF 目标文件。

## 如何复现

先核对环境与固定源码：

```bash
python3 -B tools/sync-environment.py check
```

选择一个尚不存在的输出目录，运行完整 capture：

```bash
.venv/bin/python -B research/jax-stack/matmul_probe.py \
  --output artifacts/jax-stack/cpu-matmul-new
```

脚本拒绝覆盖已有 capture，并要求清除 ambient `XLA_FLAGS`。原始产物默认在 Git 外。
当前审查对象是 `artifacts/jax-stack/cpu-matmul-003/`，包含 1,062 个被 manifest 登记的
文件，共约 3.10 MB。运行如下命令重新核对来源、全部产物字节和数值参考：

```bash
.venv/bin/python -B research/jax-stack/build_source_index.py
.venv/bin/python -B research/jax-stack/verify_research.py
```

核对其他新 capture 时给 verifier 传 `--capture`。默认 verifier 同时检查当前加载
二进制文件与 capture 指纹一致；环境迁移后应区分历史 capture 校验和当前运行环境。

用本仓库环境打开 Notebook：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/jupyter lab research/jax-stack/matmul-lowering.ipynb
```

## 哪个入口能看到什么

| 入口/文件 | 能观察到的内容 | 边界 |
|---|---|---|
| `jax.make_jaxpr(f)` / `jaxpr.txt` | 变换后的 JAX 程序表达 | 不是 backend HLO pass dump |
| `lowered.as_text('stablehlo')` | JAX lowering 后的 MLIR；本例含 StableHLO 与空 mesh 的 sdy 表示 | 空 mesh 不证明进行了多设备传播 |
| `lowered.as_text('hlo')` | 经导出接口转换得到的 HLO | 不等于实际后端逐 pass 的结果 |
| `xla-dump/*.before_optimizations.txt` | 实际 wheel 进入 HLO 优化前的模块 | 与固定 C++ source 有 VERSION-SKEW |
| `xla-dump/*.NNNN.*.txt` | 实际记录的 HLO pass 边界 | 受 flags、过滤和后端分支控制 |
| `compiled.as_text()` | 已编译对象的优化后 HLO 表示 | 不是可移植的 executable 序列化 |
| `*.mlir-passes.log`、`*.ll`、`*.o` | 某些 CPU fusion kernel 的 MLIR lowering、LLVM IR 和对象文件 | 不保证每个 HLO 操作都生成独立 `.ll/.o` |
| `executable.bin` | 本样本的可执行序列化包 | 仅重新加载自己生成且验证来源的包 |

三类 dump 的控制入口分别是：

```text
--xla_dump_hlo_as_text
--xla_dump_hlo_pass_re=.+
--xla_dump_emitter_re=mlir-fusion|llvm
```

第一轮使用 `.*` 得到 234 个文件。核对固定源码的
[`HloPassPipeline`](../../upstream/xla/xla/hlo/pass/hlo_pass_pipeline.cc#L258)
后发现这个字面量会跳过未改变 HLO 的 pass dump，改成 `.+` 才能避免该特判。
第二轮同时加入 executable 序列化验证，得到 1,044 个文件。第三轮增加 emitter dump，
补到 1,062 个文件，其中有六份 LLVM IR、三份 MLIR pass 日志和三个对象文件。
前两轮保留原始生产脚本和指纹，未改写成第三轮结果。

## 当前观察与后续问题

1. 共享 `W` 的 vmap 样本在 StableHLO 中是一个高秩 dot_general；该 wheel 的优化后
   HLO 将前两维组织成 `[12,8] @ [8,6]`，再恢复 `[3,4,6]`。这与“发射三次独立 matmul”
   不是同一个结论。具体 IR 见 `vmap_matmul/`。
2. 优化后 HLO 含 `__ynn_fusion` 标识；求导样本另外包含 multiply/copy 等融合。
   这是当前 CPU wheel 的观察，不表示已经完成指定业务的两算子 fusion 或 Pallas 注入。
3. 基础 matmul 的静态 cost 给出 384 FLOPs、416 bytes accessed；数值符合本例
   `2×4×8×6` 与输入/输出数组字节数的口径。它们不是实测 DRAM 流量或硬件利用率。
4. batch 梯度样本的 compiler memory analysis 给出 608 bytes temporary storage；
   buffer assignment 中可继续追踪 offset、复用和 liveness。这不是进程或设备的
   实测峰值，不能据此宣布完成 split 降峰值目标。

本节原始 matmul 执行证据为 `RUN-CPU + VERSION-SKEW`。新增的 typed Pallas interpreter
另记为 CPU 上的 `SIM-TPU`；生产 Mosaic lowering 也只在 CPU 主机运行，未调用 TPU backend
编译。`COMPILE-TPU`、`RUN-TPU`、匹配源码构建/加载、实际业务 Hack 等完整验收仍未完成。
