# 匹配固定源码的 CPU 基线已通过

构建 `kickoff-cpu-source-003` 已在固定 Docker 镜像中完成安装、导入、Git identity 和
已加载 native payload 核对。完整 matmul、fusion/memory、overlap 的 **18 组数值对照及
3,898 个产物**通过验证，证据为 `RUN-CPU`，无 `VERSION-SKEW`。结构化结果见
[source-runtime-results.json](../tools/source-runtime-results.json)。宿主机 `.venv` 仍保留旧 wheel。

## 构建与加载身份

JAX revision 为 `5832e866449a41c3eea6333416528039119a0fde`，XLA revision 为
`496bd4bd49db9ecbffd85da630b49c860b724604`，均无源码补丁。LLVM、StableHLO、Shardy
基础归档和 XLA 自带补丁的审计见 [source-build.md](source-build.md)。

003 wheel 为 `jaxlib-0.11.2.dev0+selfbuilt-cp312-cp312-manylinux_2_27_x86_64.whl`，
88,323,271 bytes，SHA-256 为
`8e0e4f9c87de126cd1316e46897b53368744056a1f14c7208c6061773b9b1cef`。
其 `_git_hash` 等于上述 JAX revision。31 个 `.so` 成员与首次成功构建 002 完全相同，
本次补齐了打包的 Git metadata；Bazel 报告 20,774 个 action cache hit、2 个 process，
耗时 75.418 秒。这是本次构建记录，不是应用性能测量。

构建与验证使用同一个固定镜像：
`sha256:abad11bf00a9611382e946ab243d7e3d105a4019ba3cf6bd542223debb18d6e8`。
当前 canonical manifest 验证器会读取本机 Clang，因此安装与复查在该镜像中执行。
源码和原基础 venv 只读；独立 venv 与新产物保存在
`artifacts/jax-stack/source-runtime-002`。没有把 002 的缺失 Git hash 或宿主机拒绝记录
改写成成功，原记录仍可查。

实际加载的 jaxlib `.so` 文件逐一与 wheel 成员比较大小及 SHA-256，至少包含
`_jax.so` 和 `libjax_common.so`；Python JAX 从固定 editable 源码导入。version/Git hash
检查、完整构建 manifest 验证和 native 字节绑定共同构成该基线的身份依据。

## 运行与复验

| Capture（source-runtime-002 下） | 数值案例 | 被验证产物 | 结果 |
|---|---:|---:|---|
| suite/matmul | 4 | 1,073 | 前向、vmap、grad、jit/grad/vmap；序列化同进程重载通过 |
| suite/fusion-memory | 11 | 2,043 | fusion、donation、外部 NumPy view、allocation/liveness 对照通过 |
| suite/overlap | 3 | 782 | 两个逻辑 CPU 上的 sync、async、sync-control 对照通过；另记录四个历史问题 |

输入、NumPy float64 参考和容差沿用对应生产脚本。每个 verifier 在固定镜像里重新核对
原始文件、数值、所选 IR 约束和 build/native 绑定。完整结果保存在
`suite/matmul.verified.json`、`suite/fusion-memory.verified.json`、
`suite/overlap.verified.json`，指纹由上面的结构化结果引用。

[source_runtime_suite.py](../tools/source_runtime_suite.py) 依次采集并验证三个主题；某项失败时
保留错误并继续其他独立主题，只有全部通过才返回成功。实际 Docker 参数见
`source-runtime-002/suite-launch.json`。在同一镜像、同样只读挂载及独立产物目录下，使用
已经验证的 Python 执行：

```bash
artifacts/jax-stack/source-runtime-002/baseline-env/venv/bin/python -B \
  research/software-stack/tools/source_runtime_suite.py \
  --output artifacts/jax-stack/source-runtime-002/suite-new \
  --jaxlib-build-manifest manifests/build-fingerprints/kickoff-cpu-source-003.json
```

输出目录必须是新目录，并处于该容器的可写挂载内。此命令不应直接当作宿主机 canonical
验证命令使用；宿主机 Clang 条件不同的拒绝已保留。

## 本次确认或改变的观察

1. 默认与禁用 algsimp 两组的 generic algsimp 事件均为 3；constant_folding 为 2，
   layout-assignment 为 1。三次 warm 执行没有新编译事件，自定义 leaf marker 均为 0。
   与固定源码中“generic TraceMe 位于 filter 之前”的位置一致，不能据这 3 个事件断言
   被禁用的 pass 执行了 3 次。后续自定义补丁已验证 RunHelper 的实际区间，见 [Hack 验收](../compiler/pass-event-acceptance.md)。
2. reduction 的诊断差异在匹配源码环境仍存在：打印区间按 inclusive 解释得到的 logical
   max 是 time 3 / 8,324 bytes；报告选中的 peak 位置为 time 4，所列值共 4,232 bytes。
   两者都不能直接当作物理峰值；[内存说明](../performance/fusion-and-memory.md) 的度量边界继续适用。
3. 三个 Python/MLIR 问题仍在：VMA 不匹配、未检查的 psum 没有 `.done()`、Future 默认
   layout 无效。显式 layout 加两条控制依赖后，native 错误从旧 wheel 的删除 live
   instruction 失败，变为 `NOT_FOUND: No registered implementation for untyped custom
   call to all-reduce-start for Host`。不能把该结果称为异步控制支持已修复，更不能推广到 TPU。

## 后续门槛

无补丁基线已经就绪。自定义 pass 补丁已完成 25 个 C++ 测试、patched wheel 的实际
构建与加载、默认/过滤/warm 及数值对照，再恢复源码并以新环境加载本基线进行回滚检查，
见 [完整 Hack 验收](../compiler/pass-event-acceptance.md)。原始上游树和当前独立 clone 均干净。

这份基线不增加 TPU 编译、TPU 运行、LLO、设备时间或真实推理模型的证据；完整 kickoff
及 U01–U03 仍按 [PLAN.md](../history/PLAN.md) 与 [status.json](../history/status.json) 继续。
