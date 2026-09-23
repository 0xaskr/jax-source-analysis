# SGLang-JAX：四芯片 TPU v7 的真实推理实验

用户已指定 SGLang-JAX、TPU v7、四卡、Falcon；“不修改源码”限制业务模型代码。
JAX/XLA 的可逆修改与编译/lowering 注入属于本次研究范围。模型选择和运行时版本调查
由研究执行，不再等待 U01–U03 重复确认。这里先固定可复现的业务合同；未通过的验收不记完成。

## 场景选择

| 候选 | 固定 checkout 的证据 | 本轮安排 |
|---|---|---|
| Qwen3-8B，BF16 dense | Qwen cookbook 给出 v6e-4 实测配方；本地存在完整模型实现 | 首选：长输入 prefill 的 MLP，先 TP=8、batch=1 |
| Qwen3-32B，BF16 dense | 同一 cookbook 有 v6e-4 配方 | 8B 验收后增加模型/中间激活压力 |
| MiMo-V2-Flash，MoE | cookbook 有 v7x-8 单机四芯片配方 | 保留备选；首轮不增加专家并行和量化变量 |

Qwen 在 v7 上适用是依据源码形状与拓扑作出的待验证判断，**v6e 配方不等于本项目 v7 验收**。
Google 的 [TPU7x 文档](https://docs.cloud.google.com/tpu/docs/tpu7x) 说明：
`2x2x1` 是单机四颗物理芯片，每颗暴露两个 JAX device，因此本实验使用八个 device。
Falcon 沿用现有 v7x 模板：`replica=1, device_count=8, device_topo=2x2x1`。
实际租约必须再确认四颗芯片；不以配置请求代替分配结果。

## 固定来源与版本边界

- 业务来源：[sgl-project/sglang-jax](https://github.com/sgl-project/sglang-jax/tree/338c6622cfe929223c677e641a97ea62525720b7)，
  commit `338c6622cfe929223c677e641a97ea62525720b7`；本地只读 checkout
  `/home/askr/work/kda/sglang-jax` 与该公共 commit 一致，工作树干净。
- 模型：[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218)，
  revision `b968826d9c46dd6066d109eabc6255188de91218`；下载时绑定该 revision，保存 safetensors 索引、权重文件及 tokenizer 哈希。
- 配置：hidden=4096、intermediate=12288、36 层、32 query heads、8 KV heads、head_dim=128，
  原生 max_position_embeddings=40960。本轮长度不依赖额外 RoPE 扩展。
- `python/pyproject.toml` 固定 `jax[tpu]==0.8.1`；PyPI 元数据要求 `libtpu==0.0.30.*`。
  预检显式安装 JAX/jaxlib 0.8.1、libtpu 0.0.30，采集实际设备、版本和 native SHA-256。
  这与本研究固定源码栈（JAX `2d66622450e2` / XLA `dcf304bc5dca`）不一致，
  全部增加 `VERSION-SKEW`。
  SGLang 兼容运行栈与源码研究栈分别建环境，不能用前者宣称已加载固定源码补丁。

## 业务入口与输入

全模型入口使用原生 `python -m sgl_jax.bench_one_batch` 或 `python -m sgl_jax.launch_server`，
模型路径指向固定 revision 的本地快照。业务代码及权重保持原字节；包装脚本只组织输入、
环境配置、编译观测和结果采集，不替换 `Qwen3MLP.__call__`。

基线参数：TP=8、DP=1、BF16、batch=1、input tokens=512/4096/8192，output tokens=32；
后续扩展 input=16384/32768、batch=4。服务配置固定 chunked-prefill-size=8192、
page-size=128、max-running-requests=8、context-length=32768、关闭 radix cache。
这些是选定的实验参数，正式运行前由该 commit 的 CLI 实际解析确认；性能没有预先保证。

参考输入分两组：固定 seed=20260915 的合法 token ID 张量用于形状/压力对照；中文摘要、
英文问答、算术与代码补全四类固定文本用于真实 tokenizer + greedy 生成回归。
每组保存完整 token IDs、长度、attention 相关元数据及 SHA-256。
基线与候选逐项复用输入、checkpoint、dtype、mesh、KV 容量和服务设置。

## Fusion 与 split 修改点

源码：`python/sgl_jax/srt/models/qwen3.py:158–212` 的 `Qwen3MLP`，
`python/sgl_jax/srt/layers/linear.py` 的 `LinearBase.__call__`。
当前表达是三个独立 LinearBase：gate、up、down；不是合并的 gate_up 权重。

```text
X[T,4096] @ Wgate[4096,12288] -> gate[T,12288]
X[T,4096] @ Wup  [4096,12288] -> up  [T,12288]
SiLU(gate) * up             -> mid [T,12288]
mid @ Wdown[12288,4096]     -> out [T,4096]
```

先保存实际基线 Jaxpr、StableHLO、分区后 HLO、pass dump 与 buffer assignment，确认原有
fusion，才选择新增融合点。候选包含 gate/up 的共用输入与 SiLU×up 邻域；是否值得 Pallas
替换由实际 IR、合法性和性能决定。注入位于 JAX lowering/XLA pass，必须保存补丁和关闭开关，
证明加载的是候选实现；外部重写模型函数不能算“不修改业务代码”的编译器实验。

Split 沿独立 token 维度 T，只处理 MLP，不把 causal attention 当作 token 独立。
固定服务 chunked-prefill-size 后，在编译图内部把 MLP 分块为循环；输入输出顺序、尾块、
sharding 与必要 collective 保持等价。TP=8 时 intermediate 的本地宽度为 1536。
保存分块前后完整生命周期和临时 buffer 统计，不能把服务调度的分段 prefill 计为 pass split。

## 验收顺序

1. **预检**：[Falcon 配置](../../deploy/falcon/sglang-v7-preflight.yaml) 内嵌
   [生产脚本](sglang_v7_preflight.py)。运行最多 30 分钟；确认八 device/四 chip，八设备普通
   matmul 数值、单设备 Pallas 编译/执行与 profile、原始 Qwen3MLP 的 TP=8 编译执行。
   MLP 的零输入/随机初始化仅为兼容性 smoke，不能算真实模型精度。
2. **真实基线**：下载固定 Qwen3-8B 权重，原生入口成功推理，保存基线输出、编译产物及 trace。
   先取得可重复运行数据，再改编译/lowering。编译时间和稳态执行时间分开。
3. **精度门槛**：候选与同栈基线的 MLP 输出无 NaN/Inf，归一化 RMS error ≤1e-2、
   max error / max(1,max|reference|) ≤2e-2；另保存选定子图 FP32 对照。全模型固定输入的
   greedy 32-token 输出必须完全一致；若不一致，保存首个分歧及 logit margin，不能自行放宽门槛。
   这些是本实验选择的回归门槛，不是上游官方精度承诺。
4. **内存/性能**：匹配静态临时 bytes、runtime HBM/峰值和 profiler 时间。
   KV pool、输入/权重、XLA 预分配与编译临时量分别记录；单次采样 memory_stats 不当作瞬时峰值。
   在最大稳定 prefill 形状，split 目标临时内存下降 ≥20%，稳态中位耗时回退 ≤10%；
   fusion 至少给出数值、IR、临时流量及多次稳态耗时对照，未改善也如实报告。
5. **事件**：host、compiler pass、device 三类分别验收；Pallas trace 文件存在不代表内部
   event 已在设备 plane 出现。必须解析 raw XSpace 的 device plane/core、名称、配对与次数。
   TPU LLO 仅在实际取得且识别对应产物后报告，不将 Mosaic MLIR 改名为 LLO。

预检结果、失败与真机资源状态在 [status.json](status.json) 记录。原始 captures 和集群返回
保留于忽略的 artifacts；只发布必要且脱敏的实验标识、哈希和结果。

## 当前实验结果与入口兼容性

[预检结果](sglang-v7-preflight-results.json) 已取得真实 TPU7x 身份：4 组物理坐标、每组两个
core_on_chip，JAX/jaxlib 0.8.1、libtpu 0.0.30。普通 matmul 八 device 的 producer 最大
绝对误差均为 2.634e-9；保存的 Pallas 输出独立 NumPy 复查为 2.882e-9。
原始 XSpace 无 errors/warnings；TPU:0 有五个 module 和五个 custom-call op，内部 named
regions 为零。原生 ProfileData visitor 的设备事件总数也为十；其限量输出不作为完整事件列表。

真权重加载过程中发现该 commit 的调用端缺陷共**三处**（不是两处）：

1. `bench.load_model` 漏传 `dp_size`，而 `ModelRunner.__init__` 要求该参数。
2. standalone benchmark 绕过 `TpModelWorker`，没有调用
   `ReqToTokenPool.init_cache_loc_host_buffer`。
3. `bench._run_forward_and_sample` 按两项解包 `model_runner.forward(...)`，而
   `ModelRunner._forward` 返回三项 `(output, cache_miss_count, layers_topk_ids)`。
   该 helper **无法运行**；同文件的 `latency_test` 还带着 `# TODO: Fix this function`
   且从不维护 `output_ids`。

[外部基线驱动](sglang_v7_baseline.py) 是为前两项写的兼容包装，但它在 decode 分支使用了
尚未赋值的 `ids_cpu`，**从未跑通过 decode 循环**，且缺少原生 benchmark 各调用点都有的
`process_allgather`。因此改用 [原生路径驱动](sglang_v7_driver.py)：直接调用 benchmark
所依赖的同一批原语（`ScheduleBatch.init_new`、`prepare_for_extend`、`prepare_for_decode`、
`ForwardBatch.init_new`、`LogitsMetadata`、`SamplingMetadata`），自身完成三项解包并补齐
gather；`cache_loc` host buffer 改为按 `CompilationManager` 定尺寸，并复现
`TpModelWorker` 对 `max_running_requests` 的三约束 clamp。业务文件与权重保持原字节，
这些调用端修正不计为编译器 fusion/split。

用户提供的 [XProf .pb 文档](https://outline.infiscale-tech.com/doc/xprof-pb-n7E4kJFKlZ)
revision 27 指出两个 libtpu 初始化 flags。实际 libtpu 0.0.30 拒绝第一个 flag，错误为
`Unknown command line flag 'xla_enable_custom_call_region_trace'`。
因此业务基线保留 0.8.1/0.0.30，内部事件另用 PyPI 上最新的已发布组合
**jax/jaxlib 0.11.1 + libtpu 0.0.46** 的独立环境（该组合与已换 pin 的研究源码栈同为
0.11.1，见 [交接文档 4.1](AGENT-HANDOFF-2026-09-15.md)）；
[设备采集脚本](tpu_device_events_probe.py) 在 fresh process 做默认/开启对照，
[raw XSpace 验证器](verify_tpu_device_capture.py) 区分 module/op/TraceMe、整型 ps、
oneof 与 metadata 所属 plane。两套运行都带 VERSION-SKEW；完整设备事件验收等待对照结果。
