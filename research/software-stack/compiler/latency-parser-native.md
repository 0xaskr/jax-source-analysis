# latency_metadata：原生解析器与数值边界

固定 XLA `496bd4bd49db9ecbffd85da630b49c860b724604` 的原生 C++ 测试已通过：先运行
未修改的 `LatencyEstimatorTest.GetLatencyFromMetadata`，随后加入 14 个参数化样本，
共 15 个测试全部通过。结果见 [latency-parser-results.json](../tools/latency-parser-results.json)。
生产源码没有修改；独立 clone 中的测试补丁已回滚并恢复原始字节。

证据是 `RUN-CPU / native-cpp-unit-test`。它直接调用原生
`LatencyEstimator::GetLatencyFromMetadata`，不经过 Python 对该方法的替代实现。
测试构造 HloInstruction 并读取属性，没有执行 custom-call、GPU NodeCost consumer、
真实调度或 TPU 程序。

## 从原生结果确认的行为

| 输入字符串 | 测试设定 cycles/µs | 返回 |
|---|---:|---|
| 属性缺失 | 1 | nullopt |
| `30000` | 1 | 30 cycles |
| `0` | 1 | 0 |
| `-1000` | 1 | −1 cycle |
| int64 最大值 `9223372036854775807` | 1 | 有值，约 9.223372036854776e15 cycles |
| int64 最小值 `-9223372036854775808` | 1 | 有值，约 −9.223372036854776e15 cycles |
| `9223372036854775808` / `-9223372036854775809` | 1 | nullopt |
| `30000.5`、空串、`slow`、`3e4` | 1 | nullopt |
| `30000` | 2 | 60 cycles |
| `1` | 1 | 0.001 cycle |

换算遵循 `latency_ns × CyclesPerMicrosecond() / 1000`；测试设定的 1 或 2 cycles/µs
只用于检查单位与比例，没有测量硬件频率。int64 边界值转成浮点 TimeCost 后存在舍入，
表中结果不表示保存了每一位整数精度。

解析器不校验非负性，因此零与负整数都返回值。这与
[CPU 标签传递实验](../source/source-metadata-baseline.md) 是两个环节：后者证明字符串能留在 HLO，
这次原生测试才实际检查这些字符串如何被解析。消费者是否调用该方法以及如何进入
调度，仍由 [模型选择与调用链](../performance/latency-model.md) 的条件决定。

## 源码、测试和回滚证据

- 原方法位于 [latency_hiding_scheduler.cc](../../../upstream/xla/xla/service/latency_hiding_scheduler.cc#L376)，使用 `absl::SimpleAtoi` 解析 int64。
- 原测试位于 [latency_hiding_scheduler_test.cc](../../../upstream/xla/xla/service/latency_hiding_scheduler_test.cc#L158)，覆盖有效值、非法文本和缺失属性。
- [边界测试补丁](../tools/xla-latency-parser-boundaries.patch) 只修改该 `_test.cc`；新增样本及实际返回值写入 GTest XML properties。
- 两次运行保留源码快照、完整 Docker/Bazel argv、日志、XML、测试二进制及 SHA-256。原始目录分别为 `artifacts/jax-stack/latency-parser-native-001`、`latency-parser-native-002`。
- [验证器](../tools/verify_latency_parser.py) 重新解析 15 项测试及 14 个逐例观察，校验二进制、源码、产物清单，并在临时副本重新应用和反向应用补丁。

测试使用与编译 pass 实验相同的固定镜像和 test-only Googletest 依赖副本；该依赖的
两个 XLA 补丁及复放记录见 [编译事件测试说明](pass-event-patch.md)。它没有替换宿主机
wheel，也没有改动原始上游树。

## 复现入口

在已准备的固定构建镜像、隔离 JAX/XLA clone 和 Bazel cache 中，原测试使用：

```text
bazel test @xla//xla/service:latency_hiding_scheduler_test
  --test_filter=LatencyEstimatorTest.GetLatencyFromMetadata
  --override_module=googletest=<已验证的测试依赖副本>
```

复现边界样本时，在干净且没有构建使用的隔离 XLA clone 中应用发布补丁，并将 filter
改为 `LatencyEstimatorTest.GetLatencyFromMetadata:ResearchMetadataCases/*`。保持固定
镜像、Clang、Bazel 和原参数；完整 argv 保存在两份 `launch.json`。运行后反向应用同一
补丁，检查恢复后的 test.cc hash 和 `git status`。

只复查已保存证据及反例：

```bash
.venv/bin/python -B research/software-stack/tools/verify_latency_parser.py --selftest
```

`latency-model-contract.json` 保留早期 SOURCE-ONLY 快照及当时未执行测试的标记，供历史
CPU 标签 capture 继续复查；这次实际执行使用新的结果文件，不回写旧记录。后续还需要
目标模型/后端的 consumer、调度及实际性能证据。
