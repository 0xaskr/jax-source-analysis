# 匹配源码的属性、HLO 编辑与成本复验

固定源码 wheel `kickoff-cpu-source-003` 的复验通过：**14 组 metadata 对照、2 组 HLO
改写执行、117 个产物**，证据为 `RUN-CPU`，没有 `VERSION-SKEW`。采集进程和复查进程
的 native payload 都与同一个成功构建的 wheel 逐项绑定。结果见
[source-metadata-results.json](source-metadata-results.json)。

这是对 [属性/cost](attributes-and-cost.md) 和 [latency metadata](latency-model.md) 历史
实验的独立源码复验；旧 capture、旧结果与宿主机 wheel 均保留。
[属性 Notebook](attributes-cost.ipynb) 的 8 个代码单元已执行：前六个保留旧 CPU 路径，
第七个复查这里的源码 capture 指纹、已保存身份及全部 16 组数值结果，最后一个复查
独立的 [原生 latency parser](latency-parser-native.md) 结果。

## 本例确认的机制

| 观察 | 匹配源码的结果 |
|---|---|
| metadata context、value、call | 标签分别保留在 dot 或 call 上；Python bool/integer 被规范化为字符串 |
| 两种 grad metadata 策略 | tagged call 数为 2 / 1；梯度成本均为 792 FLOPs |
| `research_flops=999999` | 4×8 与 8×6 的 dot 仍为 384 FLOPs、416 bytes accessed |
| 直接修改 MLIR 属性 | frontend string/bool 保留；typed integer 与任意 `research.raw` 丢失 |
| native HLO 属性 setter | 字符串属性修改成功，默认 dot cost 不变 |
| HLO add→subtract | 两份 HLO 都经真实 CPU 编译执行；before−after 与 2×bias 相符 |
| TPU 外层 custom-call 的 CPU 成本读取 | FLOPs、bytes、optimal_seconds 仍为 −1，表示未知 |
| 8 种 latency 输入 | 未设置/整数/字符串/零/负数/小数/非法文本/int64 溢出均重测；所选标签作为字符串传递 |
| latency 的拥有者与默认成本 | exported HLO 为 dot，optimized HLO 为 dot 和 fusion；384 FLOPs/416 bytes 不变 |

HLO 改写前后的最大绝对误差分别为 `3.68e-8`、`7.34e-8`。metadata cost 和 latency 标签
及 lowered cost 与旧 wheel 的本例观察一致。这不能证明任意属性都不影响任意后端；
已有 GPU latency consumer 的条件仍按固定源码分别解释。

opaque custom-call 使用的是已保存的 TPU 目标外层 IR，由新的 CPU reader 分析。
它没有重新执行 Mosaic TPU lowering，也没有调用 libtpu 或运行 TPU。

## 两个进程的身份门槛

两个生产脚本新增 `--jaxlib-build-manifest`，复用
[capture_runtime.py](capture_runtime.py)。完整 manifest、wheel、Git revision 与已加载
native payload 在采集前后检查。输出记录实际 Python executable、argv、helper 快照、
build binding 和产物指纹。

两个验证器会重新调用 native HLO parser 与 CPU cost analysis，因此除验证 capture 的
二进制来源，还使用 `verify_current_reader` 绑定**当前复查进程**。否则，旧 reader 读出
相同数字并不能证明匹配源码的 native reader 已被复验。

实际负对照保存在 `artifacts/jax-stack/source-metadata-gates-001`：两个生产入口拒绝
running build；两个生产入口在生成输入前拒绝旧 wheel；两个验证入口在成本重算前拒绝
旧 native reader。后四项均报告 `jaxlib runtime revision disagrees with build manifest
JAX commit`。历史 manifest 损坏、metadata-as-FLOPs 和未执行 GPU parser 测试的反例继续通过。

## 复现

在 [固定构建镜像](source-build.md) 中使用已验证的源码基线 Python。源码、基础环境及
已保存 capture 只读；为新的 output 目录提供可写挂载，保持网络关闭：

```bash
artifacts/jax-stack/source-runtime-002/baseline-env/venv/bin/python -B \
  research/software-stack/source_runtime_suite.py --suite metadata \
  --output artifacts/jax-stack/source-metadata-new \
  --jaxlib-build-manifest manifests/build-fingerprints/kickoff-cpu-source-003.json
```

`--suite lowering` 保留原 matmul/fusion/overlap 三组，`--suite all` 运行两组集合。
单独使用验证器时指定 `--capture`；复算 native cost 的验证器应使用同一个源码环境。
宿主机 `.venv` 仍是旧 wheel，且 canonical manifest 的 Clang 条件需要固定镜像。

本次完整 argv、运行日志、六个身份拒绝记录和两份完整验证结果均保存在忽略目录；
这份材料不增加 TPU 编译、设备时间、硬件 roofline 或真实业务 fusion/split 的证据。
