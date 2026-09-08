# 远端 Codex 接力上下文

> 更新时间：2026-09-08 15:10（Asia/Shanghai）  
> 当前分支：`main`  
> 当前阶段：P1 runtime provenance 与 source-built jaxlib  
> 第一项可执行工作：`q005-native-dependency-capture`

本文保存当前会话中已经确认、但不能仅靠聊天记录传递的执行上下文。机器可读状态仍以
[`manifests/status.json`](manifests/status.json)、[`manifests/baseline.json`](manifests/baseline.json)
和 [`manifests/coverage.json`](manifests/coverage.json) 为准；长期范围和验收标准以
[`PLAN.md`](PLAN.md) 为准。若本文与这些文件冲突，先运行状态校验并修正本文，不要猜测。

## 用户目标与工作边界

用户要把固定版本的 JAX → TPU 软件栈做到可搜索、可运行、可修改和可回滚的白盒状态，
面向刚进入 JAX 的 kernel 开发者。最终需要覆盖：

- `jit`、`grad`、`vmap` 的组合机制；
- Python API → Jaxpr → StableHLO/Shardy → XLA/PJRT → libtpu → TPU LLO/硬件；
- tracing、重复编译、显存、sharding、collective 和通信问题；
- JAX/jaxlib/XLA/StableHLO/Shardy/Pallas/Mosaic/libtpu/LLO 的源码修改与重编译；
- 用 pass 修改真实模型的计算图拓扑，并验证 forward、gradient、sharding、alias、
  donation、effects、cache 和性能；
- 源码导读、完整架构图、实验集、微型 JAX、分析文章与反向查询索引。

Tokamax 和自研框架只作为固定 workload 提供者。分析停在它们的纯 JAX/Pallas 边界，
不展开 JAX 以上的框架内部。当前工作区尚未取得 Tokamax、自研框架、libtpu 源码或 TPU
设备；CPU 可以使用，长时间本地构建可以接受。假设以后能够取得匹配的 libtpu 源码。

用户要求：每个可独立审阅的较大里程碑通过相应门禁后，都创建 focused commit 并推送
当前 GitHub 远端。不要等多个大步骤堆积后再推送。

## 固定基线

| 项目 | 固定值 |
|---|---|
| JAX | `5832e866449a41c3eea6333416528039119a0fde` |
| JAX source version | `0.11.2.dev20260830+5832e86644` |
| XLA | `496bd4bd49db9ecbffd85da630b49c860b724604` |
| StableHLO | `7b1b15781ccbd770f50c7eef4b0c3e03834649fd` |
| Shardy | `eb23a98329aa70d991aa2d8a51a209af1f8df8fc` |
| LLVM | `ab547095ead5464dc024d66264d9b8a987f429f3` |
| Python | `3.12.3` |
| uv | `0.12.9` |
| 当前 CPU jaxlib | `0.11.1`, build `2d66622450e2c8633cda2307688ef7aa294bd6eb` |

editable JAX 源码与当前 jaxlib wheel 不匹配。所有当前 native runtime 结论必须带
`VERSION-SKEW`；CPU 结果不能外推为 TPU layout、通信、时序或性能证据。

## 已确认的架构决定

1. 控制链和编译载荷链分开记录。控制链是 JAX dispatch/cache → jaxlib/IFRT/PJRT →
   PJRT C API/plugin → libtpu runtime；普通载荷链是 transformed Jaxpr → JAX MLIR →
   StableHLO/可选 Shardy → PJRT compile → libtpu HLO passes → TPU LLO。
2. Pallas/TPU 是条件分支。内层 kernel Jaxpr lowering 为 Mosaic TPU MLIR，经 serde
   放入外层 `tpu_custom_call` 的 backend config，再随外层 StableHLO 进入 PJRT/libtpu。
3. Mosaic TPU MLIR 不是 LLO。当前公开树没有可审查的 Mosaic → LLO 实现；相关结论在
   取得匹配 libtpu、固定 target 和 compiler dump 前只能标为待 `COMPILE-TPU`。
4. 首个 HLO transformation 机制实验使用固定源码已有的
   `jax.extend.xla.register_hlo_module_transformation`。先做故意改变语义的
   `sin → cos` 探针和两条 `dot` 合并的 detection-only matcher；生产 rewrite 要等
   runtime、sharding、alias 与 donation 合同建立后完成。
5. L3 当前只验证 producer/probe/input/结构化断言/claim-linked 本地 IR 的 lineage，
   以及 IR 字节、哈希、格式和最低结构。它不表示已经语义解析 IR；parser-aware gate
   由 Q016/I004 完成。L4/L5 在对应合同落地前保持 fail-closed。

四份架构文档和 glossary 已按固定源码独立复核。regular、Pallas/Mosaic、IFRT/PJRT
C API 与 libtpu 黑盒边界没有已知 blocker。

## 已完成的 P0 基础

- 长期计划、机器可读 baseline、coverage 与恢复状态；
- topic/capture/source-index/native-binary/build/status schema；
- evidence、coverage、project-status 三类语义校验器和隔离负例；
- JAX → TPU 全栈、regular、Pallas/Mosaic、runtime/control-plane 架构骨架和术语表；
- `jit` shape tracing 的 CPU capture，包含真实已映射 jaxlib binary 指纹与
  `VERSION-SKEW`；
- source-built jaxlib 的严格 preflight、持久 manifest/attempt/log/wheel wrapper 和
  selftest。

构建 wrapper 固定 Git、Python、Bazel、Clang、source/input bytes，隔离 ambient Git、
Python 与构建环境，验证 wheel filename/tag/METADATA、ELF native payload 和 RECORD。
`runtime_validation` 当前只能是 `not-run`。

一个未经过 wrapper 的探索性 CPU build 已结束，只用于填充 Bazel cache。它的 live
state、wheel 和结果没有进入持久证据，不要把它恢复或登记为正式 P1 build。

当前 wrapper 仍使用 `--lockfile_mode=off`，也没有归档 proxy/CA 的内容身份，因此还不是
完整的外部源码闭包。状态队列已把 `p1-bazel-dependency-closure` 放在 Q005 与正式 build
之间；不要跳过该 gate。

## 下一项：Q005 动态链接与 loader resolution

目标是在当前 `VERSION-SKEW` CPU runtime 上，把“安装包内有哪些 `.so`”推进到“进程实际
映射了哪些 ELF、每个 `DT_NEEDED` 解析到哪一个对象”。建议按以下合同实现：

1. 将 `native-binaries.schema.json` 升级到 1.1，在现有 package inventory 上增加
   `loader_resolution`。保留旧字段的语义，更新 baseline/capture/topic 和 validator。
2. 给 `labs/001-jit-cpu/probe.py` 增加显式 loader capture 开关；stage 必须仍是 `run`，
   manifest 必须绑定 producer argv、输入、stdout、native inventory 与 tool identity。
3. Linux 实现读取 `/proc/self/maps`，以 device/inode 去重实际映射的 ELF。通过打开的 fd
   做 `fstat`、大小和 SHA-256；拒绝 deleted、非普通文件、路径逃逸或读取期间发生变化的
   对象。
4. 固定 `/usr/bin/readelf`，记录其 path/version/size/SHA-256；使用 `LC_ALL=C` 和
   `readelf -hW/-dW /proc/self/fd/<n>` 读取 ELF header、SONAME、NEEDED、RPATH/RUNPATH。
   不调用 `ldd`，因为它不是当前进程 loader 已选结果的可靠证据。
5. 以所有已映射 jaxlib `.so` 为 roots，构造传递 `DT_NEEDED` closure。每条 edge 必须唯一
   解析到同一 `/proc/self/maps` inventory 中的对象；拒绝 unresolved、ambiguous、带 `/`
   的 NEEDED、basename/SONAME 冲突和 closure 外伪造节点。
6. 拒绝会改变装载结果的 ambient 环境，至少包括 `LD_LIBRARY_PATH`、`LD_PRELOAD`、
   `LD_AUDIT`、`LD_DEBUG` 与 `GLIBC_TUNABLES`。producer/capture 需要记录这一环境合同。
7. selftest 用 Clang 构建 `libdep.so` 和带 RUNPATH 的 `libroot.so`，由 `ctypes` 实际加载；
   覆盖正例，以及 hash、SONAME、NEEDED edge、root、closure、路径、环境、tool identity、
   unresolved/ambiguous 等负例。
8. 现有探索显示当前进程大约映射 61 个 ELF；22 个 jaxlib roots 的预计传递 closure 为
   30 个对象、185 条 edge，未发现 unresolved/ambiguous。这个数字只是实现时的 sanity
   check，不要写成 schema 常量或正式结论。

Q005 验收：当前 capture 中每个 jaxlib binary 能与 loader roots 对接；传递依赖全部解析，
每个对象与工具都有真实字节指纹；baseline、topic、capture、evidence validator、coverage
validator 与 selftest 全部通过；`VERSION-SKEW` 仍保留。Q005 不提高现有 coverage depth，
因为它补 provenance，而不是补 claim-linked IR 或 caller/callee。

完成后更新 `PLAN.md` 与 `manifests/status.json`：关闭
`q005-current-runtime-provenance`/`q005-native-dependency-capture`，把
`p1-bazel-dependency-closure` 变成第一项可执行动作，然后单独 commit 并 push。

## 后续 P1 顺序

1. 归档并校验 Bazel module graph、resolved repositories、registry/module extension
   输入和下载完整性；选择可审查的 lock error/update 重放策略。
2. 从该 committed checkpoint 启动第一个 wrapper-managed official build，保存 manifest、
   durable log 和 wheel SHA-256。
3. 在隔离环境安装 source-built wheel，记录匹配 runtime provenance，并跑 Lab 001。
4. 加入一个默认关闭的 native diagnostic patch，重编译、证明实际加载、回滚。
5. 创建通用 Lab 模板，再进入 transformation 和 pass 系列实验。

## 恢复和验证命令

新会话先执行：

```bash
.venv/bin/python -B tools/project-status.py --check
.venv/bin/python -B tools/project-status.py
git status --short
git -C upstream/jax status --short --untracked-files=all
git -C upstream/xla status --short --untracked-files=all
```

当前 P0 提交前已经通过：

- build wrapper selftest（24 个正负合同输出）；
- evidence selftest（44 个 evidence 负例、4 个 baseline 负例）；
- coverage selftest（25 个隔离用例）；
- project-status selftest（20 个负例）；
- live evidence：1 topic、1 capture、34 个文件哈希；
- baseline verify，结果为 `VERSION-SKEW`；
- 严格 jaxlib build preflight；
- Draft 2020-12 schema 检查、Markdown 本地链接/代码围栏检查和 `git diff --check`；
- `uv lock --check --offline`；
- 102 个登记 submodule（85 initialized、17 lazy）和 1 个精确源码包校验。

editable JAX 会在普通 Python 启动中产生 ignored `.pyc`，而 live-source evidence 和正式
build preflight 会按设计拒绝它们。运行 capture、baseline verify 和提交前门禁时使用
`python -B`；正式 build 前显式清理 `upstream/jax` 下的 `.pyc/.pyo`，然后先跑：

```bash
.venv/bin/python -B tools/validate-evidence.py --require-live-source-state
.venv/bin/python -B tools/check-jaxlib-build-env.py --strict
```

不要为了让门禁变绿而放宽 bytecode、revision、hash、路径或 loader 环境检查。
