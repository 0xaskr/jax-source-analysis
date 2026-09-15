# JAX 软件栈研究交接上下文（2026-09-15）

> 接手目标：继续现有研究，不从头重做。先阅读本文件、`PLAN.md`、`status.json` 和具体失败日志。
> 用户最新要求是“输出详细上下文到本地，交给其他 agent”。本次交接期间只查询状态，不再启动新实验。
> **最新实验 baseline-005 已失败，最新租约查询为 `[]`；没有需要沿用的本任务活动 TPU 租约。**
> **业务 fusion/split 和内部设备命名事件尚未验收完成。不要把成功采集日志当成已经解析的设备事件。**

## 1. 用户已经确认的目标和边界

当前工作目录：`/home/askr/github/jax-source-analysis`。

用户最初要求清除旧分析文档、实验记录，包括当时暂存的新 `docs/kickoff.md`，但保留源码。
清理已在工作区和暂存区体现。后来开始的是清理后的新 kickoff 研究，不应恢复旧项目状态文件。

本轮最重要的用户原话：

- “先阅读文档，有什么不确定的，直接问我”。
- XProf 范围：“三类都研究：host、编译 pass、设备执行”。
- 业务/设备：“有没有什么 TPUV7 上能跑的，4卡能跑的场景，可以找 sglang-jax 去里面找，限制业务模型代码，falcon”。
- 多次对旧的三项输入问题回复 `continue`。这些是要求继续执行，**不要继续重复询问 U01/U02/U03**。
- 最新补充：内部设备事件 `.pb` 解析参考 `https://outline.infiscale-tech.com/doc/xprof-pb-n7E4kJFKlZ`。

落实方式：

1. SGLang-JAX 是真实业务提供方，具体模型/输入/回归门槛由 agent 选择。
2. 使用 Falcon 的 TPU v7 四颗物理芯片。TPU7x 每芯片两个 JAX device，因此是一台机器、8 个 device、`2x2x1`。
3. 不修改业务模型源码。允许可逆的 JAX/XLA 修改、编译/lowering pass 或 Pallas 注入。
4. 外部实验驱动可以组织输入、初始化和取证；不能替换模型计算函数后宣称完成编译器 fusion/split。
5. 真实业务、CPU 参考、TPU 编译、TPU 执行、XSpace 离线解析分别记证据级别。
6. 本轮已选择 **Qwen3-8B 长输入 prefill 的 MLP** 为首个场景；32B 是后续压力扩展，MiMo-V2-Flash 是备选。

完整合同见 [sglang-v7-workload.md](sglang-v7-workload.md)。其中门槛是实验选择，不是用户逐个指定的参数，也不是上游官方精度承诺。

## 2. 权威计划、文档和证据约定

新 kickoff：

- URL：`https://outline.infiscale-tech.com/doc/research-plan-jax-kickoff-ib5QULKSS4`
- Document ID：`a36682df-e271-4500-9325-6c0ab7c47692`
- revision **51**，更新时间 `2026-09-14T11:26:38.696Z`
- 正文：`artifacts/jax-stack/kickoff/revision-51.md`
- SHA-256：`7f54e8d6f7316aa57cf2f22ee40a6082f7dcb7dbb056831219d87bf168d56a0c`
- 来源记录：[kickoff-source.json](kickoff-source.json)

活跃状态是 **`research/jax-stack/PLAN.md` 和 `research/jax-stack/status.json`**。
根目录 `HANDOFF.md`、`PLAN.md`、`manifests/status.json`、`manifests/coverage.json` 和旧 evidence 文档已按用户指令删除。
根目录 `tools/project-status.py --check` 和无参数恢复摘要因缺少旧 status 而失败，这是已知清理结果，不能据此重建旧队列。
`manifests/baseline.json` 仍在，但它记录的是早期 CPU 环境，不能当作当前 TPU 身份。

新 PLAN 覆盖 R01–R12：组件/API、matmul lowering、普通/Pallas、可逆编译 Hack、fusion、内存、split、overlap、属性、roofline、自定义 cost、三类事件。
CPU/source 子项已经做了很多；不要用更多教学 CPU 示例代替剩余的真实业务/TPU 验收。

证据术语：`SOURCE-ONLY`、`RUN-CPU`、`SIM-TPU`、`COMPILE-TPU`、`RUN-TPU`、`REPLAY-OFFLINE`。
运行二进制与研究源码不匹配时必须带 `VERSION-SKEW`。**Mosaic TPU MLIR 不是 LLO**。
原始/private/大产物保留在忽略的 `artifacts/jax-stack/`；只将必要且脱敏的研究材料纳入 Git。

本轮没有启动子代理；当前约束是不在用户或适用说明明确要求前自行委派。本次“交给其他 agent”是交接请求，不是要求当前自动启动新 agent。

## 3. Git 状态：必须保留的用户改动

交接时重新核对：

```text
root HEAD: a8e2a90ccff86cf98b56970cbbb9dcfce1ba2971
publication worktree HEAD: b28ebb6be1bd7fc4cd8ce809374326998e972bb9
origin: git@github.com:0xaskr/jax-source-analysis.git
publication worktree: artifacts/jax-stack/publication-worktree-001
```

根仓库有大量用户已经暂存的删除/修改，根 `README.md` 另有无关的未暂存修改。
**本轮开始以来原暂存区二进制 diff 的 SHA-256 始终为：**

```text
49866b500a618ed6173968aca2f639bc1219d95550ba953839e51599a376a55a
```

检查方法：

```bash
git diff --cached --binary | sha256sum
git status --short
git -C artifacts/jax-stack/publication-worktree-001 status --short
```

不要执行 `git add -A`、reset、stash 整个工作区、checkout 覆盖或强推。
本轮新增的 SGLang/TPU 文件和状态改动**尚未 commit/push**；上次成功发布是 publication **020**，记录：
`artifacts/jax-stack/milestone-publication-020/record.json`。下一编号 **021**。

之前的发布方式：只提交指定研究文件（使用 `git commit --only ... -- <精确路径>`，前后验证原暂存 diff）；
在独立 publication worktree fetch 并 fast-forward 到远端，再 merge 本地研究 commit，检查 merge 只包含授权研究路径，正常 push。
不要把远端分支合并进脏的根工作区。本次还新增 `deploy/falcon/*.yaml`，发布白名单需要显式包含这些文件。
项目要求独立里程碑验证后提交/推送；当前用户转为本地交接，尚未完成本轮研究里程碑的发布。

## 4. 固定源码、CPU 构建与已有工作（无需重做）

固定版本：

| 组件 | revision |
|---|---|
| JAX | `5832e866449a41c3eea6333416528039119a0fde` |
| XLA | `496bd4bd49db9ecbffd85da630b49c860b724604` |
| StableHLO | `7b1b15781ccbd770f50c7eef4b0c3e03834649fd` |
| Shardy | `eb23a98329aa70d991aa2d8a51a209af1f8df8fc` |
| LLVM | `ab547095ead5464dc024d66264d9b8a987f429f3` |
| XProf | `68dba1826c37986af41f6119930ec315592a51f6` |

### 4.1 JAX 版本判定（2026-09-15 复核，权威结论）

**结论：继续使用当前 pinned 的 `5832e866...`。它是可获得的最新 JAX，无需换版本。**

判定依据（全部为当日实测，不是推测）：

| 来源 | 结果 |
|---|---|
| PyPI `jax` | 最新 stable **0.11.1**（2026-08-17 上传），`0.11.2` **不存在** |
| PyPI `jaxlib` | 最新 stable **0.11.1**，`0.11.2` **不存在** |
| GitHub releases | 最新为 `jax-v0.11.1`，published 2026-08-17T20:45:31Z，**无更新 tag** |
| pinned commit 日期 | **2026-08-30T08:08:43Z**，比 0.11.1 发布**晚 13 天** |
| pinned 源码声明 | `upstream/jax/jax/version.py:24` → `_version = "0.11.2"` |
| pinned 自述的兼容 jaxlib | `upstream/jax/setup.py:25` → `_latest_jaxlib_version_on_pypi = '0.11.1'` |

因此 pinned commit 是 **0.11.2 的开发快照**：版本号已 bump，但 0.11.2 尚未发布。
它比任何已发布版本都新，正是用户要求"有更新的就用最新"的那个最新版本。

**交叉一致性已核对**：pinned JAX 自身声明的 XLA 依赖就是本项目 pin 的 XLA。
`upstream/jax/MODULE.bazel:35-36` 写的是 `xla-496bd4bd49db9ecbffd85da630b49c860b724604`，
与 `upstream/xla` 的 HEAD 完全一致。所以 JAX/XLA 这一对无需调整。

**必须记录的两个版本陷阱：**

1. **不要写 "JAX 0.11.1" 来描述源码栈。** 那是 PyPI 邻近版本，仅用于 004 的独立设备事件
   环境。pinned 源码是 0.11.2 快照。源码构建出的 wheel 命名为
   `jaxlib-0.11.2.dev0+selfbuilt-...`，`0.11.1` 与 `0.11.2` 在文档里必须区分。
2. **libtpu 不要升到最新的 0.0.47。** PyPI 上 libtpu 最新已是 **0.0.47**，但
   `upstream/jax/setup.py:27` 钉的是 `_libtpu_version = '0.0.46.*'`，且已发布的
   0.0.46.1 存在。源码 jaxlib 必须配 **0.0.46.x**，用 0.0.47 会跳过 pinned 声明。
   这与"用最新"不冲突：libtpu 的版本由 pinned 源码规定，不是自由选择。

**关于 SGLang 业务栈 0.8.1**：那是 `python/pyproject.toml` 的硬性要求，不是版本选择，
因此**不升级**。研究源码栈（0.11.2 快照）与业务栈（0.8.1/libtpu 0.0.30）继续分开记录，
两者都带 `VERSION-SKEW`，只有源码栈能在修复后摆脱它。

本轮起始检查五个 pinned upstream 工作树均干净；交接时再次检查 JAX/XLA 和外部 SGLang 树仍干净。

已有 source-bound CPU 构建：

- build ID：`kickoff-cpu-source-003`
- manifest：`manifests/build-fingerprints/kickoff-cpu-source-003.json`
- wheel SHA：`8e0e4f9c87de126cd1316e46897b53368744056a1f14c7208c6061773b9b1cef`
- wheel 大小：88,323,271 bytes。
- 固定 Docker image：`sha256:abad11bf00a9611382e946ab243d7e3d105a4019ba3cf6bd542223debb18d6e8`
- 独立 runtime Python（在固定 image 中使用）：`artifacts/jax-stack/source-runtime-002/baseline-env/venv/bin/python`
- 独立构建 clone：`artifacts/jax-stack/source-build-001/clones/{jax,xla}`，之前补丁已回滚。
- 宿主 `.venv` 仍是 PyPI jaxlib 0.11.1、JAX 源码 editable 的组合；不要用它冒充 source wheel 003
  （003 是自建的 `jaxlib-0.11.2.dev0+selfbuilt-...`）。
- Clang 18.1.3 身份验证依赖固定 image，宿主没有匹配工具链。

以下为已保存的历史验收摘要，本次没有全部重新跑；以对应文档、结果 JSON 和 capture 为准：

| 已完成子项 | 已有证据 |
|---|---|
| 匹配源码 CPU 基线 | 18 组数值、3,898 个产物；source-runtime-baseline.md/source-runtime-results.json |
| CPU 编译 pass Hack | 25 个 C++ 测试，实际构建/加载，cold/warm/filter，源码和 runtime 回滚；pass-event-acceptance.md/pass-hack-results.json |
| pass marker 修正 | 原 `program_id` 被 JSON 导出隐藏；增加 `research_program_id`；默认/过滤自定义事件 127/124，algsimp 3/0，generic 仍 3/3 |
| metadata/cost/HLO 编辑 | 14 组 metadata + 2 组 HLO 改写，117 个产物；源码 wheel 复验 |
| CPU fusion/memory | 11 组样本、2,028 个产物；不是实际业务 split 或 TPU HBM 峰值验收 |
| matmul 逐 pass | 640 个边界、584 组配对、22 组叶子变化；matmul-pass-walkthrough.md |
| CPU executable/thunk | 33 对 producer/完成事件，19 个反例、5 个 Notebook 单元；cpu-executable-and-trace.md |
| 原始 XSpace context | 49 对关联、13 对跨线程、超大 uint64/逆序/异常；20 negatives/6 positives/5 notebook cells；xspace-contexts.md |
| 普通 PJRT 公开链 | SOURCE-ONLY 的 32 个新入口、24 条关系；12 claims/15 lexical guards/9 negatives；tpu-runtime-boundary.md |
| Shardy 往返/分区 | Shardy/GSPMD 双 CPU 共 6 次数值、65 个产物；全局 [8,12] 属性传播与后续局部 [4,12] 分区分开；9 negatives、5 notebook cells |
| 来源索引 | 当前共 219 entries / 92 edges |

没有正在继续的旧 CPU build/notebook 任务。不要重新跑已通过的 broad CPU suite，除非相关代码/证据发生变化。

之前因用户输入未补齐做过三次 no-progress audit，并把外部 goal service 标为 `blocked`。
现在用户已经补齐范围，仓库状态已恢复 `in-progress`；旧 blocked audit 仅保留历史。
goal 工具只支持 complete/blocked，没有 active/resume，不要创建新 goal 来绕过，也不要因旧状态停止工作。

## 5. SGLang-JAX 和模型合同

只读工作树：`/home/askr/work/kda/sglang-jax`。

```text
HEAD: 338c6622cfe929223c677e641a97ea62525720b7
local origin: git@github.com:primatrix/sglang-jax.git
public reproducible source: https://github.com/sgl-project/sglang-jax/tree/338c6622cfe929223c677e641a97ea62525720b7
```

GitHub API 已确认同一个 commit 存在于公共 upstream，远端实验通过 codeload 下载该精确 tarball，
安装 `python/` 子目录。tarball 没有 Git metadata，所以安装版本显示 `sglang-jax==0.0.0.dev0`；
不能依据这个版本字符串定位源码，必须使用固定 SHA、归档哈希和模型源码哈希。

该 checkout 的 `python/pyproject.toml` 明确要求 JAX **0.8.1**；TPU extra 对应 libtpu **0.0.30**。
其余依赖也要记录实际安装版本，例如 Flax 0.12.4，transformers 实际解析为 4.57.6。
业务运行用这套支持栈，与研究固定 JAX/XLA **不匹配，始终 VERSION-SKEW**。

主模型：

```text
Qwen/Qwen3-8B
Hugging Face revision: b968826d9c46dd6066d109eabc6255188de91218
hidden=4096, intermediate=12288, layers=36
query_heads=32, kv_heads=8, head_dim=128
max_position_embeddings=40960, vocab=151936, dtype=BF16
```

**不要误写 intermediate=14336 或 layers=32。** 32B 扩展为 hidden=5120、intermediate=25600、layers=64。
8B TP=8 的本地 intermediate 宽度为 1536；32B 对应 3200。

公开 Qwen cookbook 的验证硬件是 **v6e-4**，不能称其已经验证 v7。MiMo-V2-Flash cookbook 有 v7x-8 四物理芯片方案，
但首轮未选择 MoE/量化的复杂场景；不要复述其可疑的“20GB/chip”估计。

源码中的真实 MLP：

```text
python/sgl_jax/srt/models/qwen3.py:158–212  Qwen3MLP
python/sgl_jax/srt/layers/linear.py        LinearBase

X[T,4096] @ Wgate[4096,12288] -> gate[T,12288]
X[T,4096] @ Wup[4096,12288]   -> up[T,12288]
SiLU(gate) * up               -> mid[T,12288]
mid @ Wdown[12288,4096]       -> output[T,4096]
```

gate/up/down 是三个独立 LinearBase，不是虚构的 merged gate_up_proj。LinearBase 使用 `lax.dot_general`，
gate/up 的权重 axes `(None,"tensor")`，down 为 `("tensor",None)`。

Fusion：先看实际基线 HLO 是否已经融合，再选择 gate/up 共用输入、SiLU×up 等候选。
Split：只沿 MLP 的 token T 维度，不把 causal attention 当独立 token；固定服务 chunked-prefill-size，
不能把 scheduler 分段 prefill 计为编译器循环 split。注入需在 JAX lowering/XLA 层，有可逆补丁和加载证据。

选定基线：TP=8、DP=1、BF16、page=128、max-running-requests=8、max-total-tokens=65536、
max-prefill/chunked-prefill=8192、context=32768、mem-fraction-static=0.4、radix cache 关闭、disable-precompile。
输入是 seed 20260915 的固定合法 IDs，长度 512/4096/8192，另有四条中英文/算术/代码文本。
每例计划生成 32 tokens，cold/warm/profile 三次；保存全部输入 IDs、prefill logits、生成 IDs 和 trace。
这只有一次未 profile 的 warm 运行，**不足以宣称稳定性能优劣**。

合同的候选回归门槛：MLP normalized RMS ≤1e-2、归一化 max error ≤2e-2、无 NaN/Inf；
全模型固定 greedy 32-token 输出完全一致；split 目标临时内存降 ≥20%、稳态中位耗时回退 ≤10%。
尚未运行候选改写，不能宣称这些指标已通过。

## 6. Falcon 使用方式与资源范围

```text
CLI: /home/askr/.falcon/bin/falcon
version: 0.1.4, commit cacf30d626284f0d100f30c76d1075c0d14ae000
cluster: tpu-training-antgroup
cluster_id: cl-v5l8pq8yso
topology: v7x, 2x2x1, replica=1, device_count=8
physical chips: 4
creator_id of this task: c9222605-1913-5c13-8c0f-4425c757adae
```

模板来源：`/home/askr/work/pallas-kernel/deploy/falcon/mock-local-tpu-v7x.yaml`。
本地 Falcon 源码 `/home/askr/work/Falcon` 的 renderer 测试也明确 `DeviceCount=8` 配 `2x2x1` 是请求四颗芯片。
资源查询出现许多其他实验；**不能 abort/exec 其他人的 Pod**，本轮只操作自己创建的明确 ID。

可用命令：

```bash
FALCON=/home/askr/.falcon/bin/falcon
$FALCON workflow exp submit -f deploy/falcon/sglang-v7-baseline.yaml --output json
$FALCON exp get EXP_ID --output json
$FALCON exp logs EXP_ID --tail 30
$FALCON lease list --job-id JOB_ID --output json
$FALCON exp abort EXP_ID --reason '具体原因' --yes --output json
```

`workflow validate` 验证的是 workflow schema，不适用于这里的 experiment YAML，曾报缺少 schema_version/workflow/steps；
这不是 experiment 配置无效。`cluster offering --allowed true` 语法不对，应 `--allowed=true`，该集群此次返回 `[]`，
不代表没有可分配资源（后续多次实际拿到了四芯片）。

远端使用 `python:3.12`，实际 Python 3.12.14；该 image tag 未固定 digest，运行身份有记录但不是完全锁定镜像。
uv 固定 0.12.9。实验顶层 `timeout --signal=TERM --kill-after=30s 45m`。
baseline 的 EXIT trap 把结果复制到 GCS mount，并 sleep 300 秒用于取回；若 trap 中 cp 失败，不能假设回传窗口一定存在。
预检为 30m 上限。不要在同一四芯片预算下并发启动多个 TPU 实验。

## 7. 全部本轮实验账本

| 尝试 | Experiment / Job | 已知结果 |
|---|---|---|
| preflight-001 | `exp-rpa9xkpbyu` / `job-ki5t2jegpi` | 普通/Pallas 已运行；外部 smoke 默认 Auto mesh 与 SGLang Explicit 要求不符，MLP 初始化失败 |
| preflight-002 | `exp-mchaxlzr62` / `job-e44jkjviqk` | 成功；本地 raw capture 已取回并独立复查 |
| baseline-001 | `exp-2cow48lqp1` / `job-ood2fljibh` | 真权重下载完成；bench.load_model 漏传 ModelRunner 必需 dp_size，失败 |
| baseline-002 | `exp-raj548h133` / `job-wovvrtriro` | 外部补 dp_size 后加载权重；缺少 cache_loc_host_buf 初始化，失败；回传后主动释放自己的等待窗口 |
| baseline-003 | `exp-c1c8fhpjx2` / `job-iibztpzpc6` | 在模型前做设备实验；libtpu 0.0.30 拒绝 region flag；未运行该轮模型，主动释放等待窗口 |
| baseline-004 | `exp-tov960pjtm` / `job-b2qxvr4cvq` | PyPI jax/jaxlib 0.11.1 + libtpu 0.0.46 的 default/enabled 四组采集日志成功（非源码构建）；业务加载 399 regular weights，forward 返回后旧 bench 按两项解包而失败 |
| baseline-005（最后） | `exp-uvj2yptpe2` / `job-pp9r06b00d` | **失败 exit 2**：打包旧 004 的 `device-events-enabled` 时 Cannot stat；还没进入模型阶段；交接查询租约 `[]` |

005 的 artifact ID：`art-6o4aeuftmz`。
最后状态实查记录：`artifacts/jax-stack/agent-handoff-20260915/{current-experiment,current-lease,current-logs}.json`，
每份含 UTC checked_at、命令、返回码和原始输出。

005 已提交并结束，不要误认为用户中断取消了远端提交。上轮未完成的本地 exec session 67742 已在本次交接中确认结束。
本次交接没有创建新的 TPU 任务。

### 7.1 已被本地验证的 preflight-002

本地成功取回目录：

```text
artifacts/jax-stack/sglang-v7-captures-001/sglang-v7-preflight-001/
artifacts/jax-stack/sglang-v7-captures-001/sglang-v7-preflight-002/
artifacts/jax-stack/sglang-v7-preflight-002/audit.json
artifacts/jax-stack/sglang-v7-preflight-002/profiledata-visitor.json
research/jax-stack/sglang-v7-preflight-results.json
```

两次预检归档：405,724 bytes，SHA `f1c1e824895468187de997da1e883ba376c6cee5b3f24da557792fc0f3c0df25`，共 28 个文件。

- runtime：JAX/jaxlib 0.8.1，libtpu 0.0.30。
- 真机 8 devices 的 coords 是四组 `(0/1,0/1,0)`，每组 core_on_chip=0/1，device_kind=TPU7x。
- 普通 matmul 八 device 的 producer 数值最大误差均 `2.6338763323141556e-9`。没有保存八份单独输出数组，不能说都做了独立输出复算。
- 保存的 Pallas 输出由本地 NumPy float64 对照复算，max abs `2.881982017616247e-9`。
- 未改动 Qwen3MLP 的随机初始化、零输入 `[128,4096]`、TP=8 smoke 成功；不是 checkpoint 精度验收。
- libtpu 已加载 native SHA：`9957eae0e4214184b2784d994bbc4d50e44ffea263de303b6461faac1dbc2859`。
- 21 个已加载 native 映射哈希均匹配记录的 package payload 哈希。
- 原始 XSpace errors/warnings 均空。
- `/device:TPU:0` 有 5 个 `XLA Modules`、5 个 `XLA Ops`；后者是 `tpu_custom_call`。
- 默认采集没有内部 `research_v7_kernel/dot/store` 区间；host 上有五次 `research_v7_step`，不能拿来替代设备 region。
- 用邻仓 `parse_xplane.py` 的 ProfileData visitor 核对设备事件总数也为 10。
  该次 `--limit 0` 有意不输出事件样本，`events_truncated=true`；完整列表以 raw protobuf audit 为准。

### 7.2 benchmark 入口漂移和当前外部驱动

固定 commit 的 benchmark 与当前内部 API 不一致，已经连续发现三处：

1. `bench.load_model` 不传 `dp_size`，而 ModelRunner.__init__ 要求该参数。
2. benchmark 绕过 `TpModelWorker`，没有调用 ReqToTokenPool.init_cache_loc_host_buffer。
   正常 worker 的初始化位置是 `srt/managers/tp_worker.py` 约 255 行，容量来自 compilation_manager cache buckets。
3. `bench._run_forward_and_sample` 按 `(logits_output, _)` 解包，当前 ModelRunner.forward 返回
   `(output, cache_miss_count, layers_topk_ids)`。KV 更新已经在 ModelRunner._forward 内 replace_all。

当前 `sglang_v7_baseline.py` 在外部显式构造 ModelRunner(dp_size=1)，初始化 host buffer 到固定 KV cap，
并使用一个 BenchModelRunner 子类，将原 forward 的三项结果在确认 `layers_topk_ids is None` 后适配为旧 benchmark 的两项。
它没有改业务模型文件，也没有修改已安装的 SGLang 源码文件；是调用端兼容包装，不能计为 compiler Hack。

**第三项适配尚未在真实模型中验证**：005 在运行模型前就因旧产物回收失败了。
004 到了 forward 返回/解包位置，但未完成驱动的 block_until_ready、完整 token/logit 保存和精度断言，
所以不应把它标为全模型基线验收成功。

接手者可继续验证这个小适配，也可选择使用正常 launch_server/Engine/TpModelWorker 入口绕开过期 benchmark。
如果继续 benchmark，应先对照当前 ModelRunner/SamplingMetadata/ScheduleBatch API，避免每次只在真机发现一个旧调用问题。

## 8. 用户提供的 XProf `.pb` 文档与设备事件现状

已通过 Outline connector fetch 阅读全文并保存，**无需再次要求用户贴文档**：

```text
URL: https://outline.infiscale-tech.com/doc/xprof-pb-n7E4kJFKlZ
ID: dab39694-9aea-49db-894d-7c744eb9c324
revision: 27
updated_at: 2026-08-24T07:03:19.423Z
snapshot: artifacts/jax-stack/xprof-pb-reference-001/revision-27.md
source metadata: artifacts/jax-stack/xprof-pb-reference-001/source.json
```

文档本身固定的是另一组 JAX/XLA/XProf revisions，不自动等于本项目 source pin。
正文只保留在忽略目录，不把私人 Outline 全文发布到 GitHub。

关键建议已经用于实验：

```text
在 import jax / libtpu 初始化前设置：
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"
```

- libtpu **0.0.30 实测拒绝** `xla_enable_custom_call_region_trace`，保留原错误日志。
- PyPI **jax/jaxlib 0.11.1 + libtpu 0.0.46** 独立 venv 接受 flags；004 日志有两个
  `TPU_REGION_CAPTURE_OK`。这**不是**源码栈：pinned 源码是 0.11.2 快照（见 4.1）。
- 本仓库 pinned `upstream/jax/setup.py:27` 也声明 libtpu `0.0.46.*`。
- 但 004 的成功采集原始对照还没成功取回，005 回收时还发现 GCS 里缺 enabled 目录。
  **原因尚未诊断；不要推测是编译器丢标记或声称该对照已经解析完成。**

解析原则：

1. 原始 `.xplane.pb` 是入口；JSON viewer 可能丢字段、裁剪事件，不能作为完整原始证据。
2. 先枚举 plane/line/event，再定义 selector；重点区分 `XLA Modules`、`XLA Ops`、`XLA TraceMe`。
3. 元数据 ID 只在所属 plane 内解释；name 与 display_name 分开。
4. `start_ps = line.timestamp_ns * 1000 + event.offset_ps`，保留整数精度。
5. `offset_ps` 与 `num_occurrences` 是 oneof；聚合记录不能当普通 timeline。
6. 检查 errors/warnings，且“完整遍历文件”不保证硬件采集无损。
7. sum、interval union、module window、自耗时和跨设备聚合不是同一指标。
8. `.pb` 中 metadata plane 可能嵌 HLO proto，后续可用于真实编译图分析；Mosaic MLIR 仍不等于 LLO。

可复用现有 parser：`/home/askr/work/pallas-kernel/scripts/parse_xplane.py`，用 `jax.profiler.ProfileData.from_file`。
本轮新 raw verifier：`research/jax-stack/verify_tpu_device_capture.py`，复用已有严格 protobuf schema工具。

```bash
.venv/bin/python -B research/jax-stack/verify_tpu_device_capture.py \
  --capture artifacts/jax-stack/sglang-v7-captures-001/sglang-v7-preflight-002/capture \
  --result artifacts/jax-stack/sglang-v7-preflight-002/audit.json
```

该命令已成功。启用对照的 `--enabled` 分支和其中三个 negative tests **尚未对真实 enabled capture 运行**。
当前它预期 `XLA TraceMe` 上三个精确名称各五次并验证嵌套。真实名称/line 若不同，要先读 raw 数据再调整明确 selector，不能伪造通过。

schema 依赖：

- `artifacts/jax-stack/xspace-context-audit-002/capture`
- `artifacts/jax-stack/cpu-thunk-schema-002/python-tool`（protobuf 7.34.0 的记录副本）
- `research/jax-stack/xspace_contexts.py` / `cpu_executable_parser.py`

先通过 `schema_pool()` 初始化记录的 protobuf，再 import google.protobuf；不要随意 pip 安装后覆盖原分析工具。
本地 verifier 用 `.venv/bin/python -B`（系统 python3 没有 numpy）；这只是数据解析，不是新 source-bound CPU/TPU 执行。

## 9. 产物回收的实际坑和可用路径

远端 mount：

```text
bucket: tops-pallas_kernel
mount: /root/tops-pallas_kernel
task prefix: jax-source-analysis/
preflight prefixes: sglang-v7-preflight-001、002
baseline prefixes: sglang-v7-baseline-001 ... 005
```

本地 `gcloud` 可列 prefix，但 object get 多次 403。已有账号是默认 Falcon service account 和 `mail@askr.dev`；
试过显式 `--account`（没有改默认配置），不要反复尝试相同命令或输出凭据。
`gcloud storage cp --recursive`、user-account rsync 均未成功取回，不能把本地空目录/0 byte 文件当产物。

`falcon exp cp` 在这里等待后报 `agent session not connected`。命令通道 `falcon exp exec` 则能工作，
但有一次 HTTP 499 client connection closing，另一次原实验已结束导致 HTTP 400。

成功拿到 preflight 归档的方式：

1. 在仍运行的自己 Pod 内直接调用 tar，读取 mount 下自己实验的已存产物。
2. 通过 `base64 --wrap=1000 archive.tar.gz` 输出到实验日志。
3. `falcon exp logs EXP_ID --tail 2000` 保存到本地文件，不把大段 base64 打到对话。
4. 提取以 `H4sI` 开头的 base64 段，decode，gzip CRC 检查，tar 路径范围检查后解包。

不要用默认 76 字符换行：服务器尾行 cap 曾截掉 gzip 的开头。
不要用 100000 字符换行：CLI command output 曾只保留前 65536 bytes。
`exp exec` stdout 有时只返回 command 状态，实际内容在 experiment logs；因此需要分开保存原始 command 返回和 logs。
`exp exec ... -- sh -c '复合命令'` 曾因 CLI 拼接丢引号而失败；单程序直接参数调用更可靠。

005 manifest 尝试在模型前自动打包并打印旧 004 的 region archive，添加 `REGION_ARCHIVE_BEGIN/END` 边界，
但因为 enabled 目录缺失导致 `set -e` 提前终止。**下一轮应把旧产物回收与业务基线解耦**，
并在设备采集刚完成、原文件仍存在时直接回收/验证，而不是依赖失败后的最终 cp。

现存目录里有失败传输留下的空 tar、日志碎片，不要拿错：

```text
有效：artifacts/jax-stack/sglang-v7-preflight-002/preflight-captures-via-logs.tar.gz
无效/失败：preflight-captures.tar.gz（曾为 0 bytes）、preflight-captures-via-command.tar.gz、
           transfer-wide.log（64 KiB 截断）、retrieval-experiment.log（缺开头）
```

## 10. 本轮新文件与验证状态

| 路径 | 用途 / 当前状态 |
|---|---|
| `research/jax-stack/sglang-v7-workload.md` | 场景、形状、注入边界和精度/内存合同；部分进展段早于最后尝试，以本交接为准 |
| `research/jax-stack/sglang_v7_preflight.py` | 成功的 002 producer：八 device 普通算子、单 device Pallas、原始 MLP smoke |
| `research/jax-stack/sglang-v7-preflight-results.json` | 已核对的简要 TPU 兼容性结果 |
| `research/jax-stack/sglang_v7_baseline.py` | 最新外部真权重驱动，包含三处 benchmark API 兼容处理；完整成功运行仍待验证 |
| `research/jax-stack/tpu_device_events_probe.py` | fresh process plain/named、每组五次 trace，保存运行版本和 libtpu native hash |
| `research/jax-stack/verify_tpu_device_capture.py` | raw protobuf/NumPy 审计；preflight 分支通过，enabled 区间/negative 分支待真实材料 |
| `deploy/falcon/sglang-v7-preflight.yaml` | 内嵌 002 producer 的 manifest |
| `deploy/falcon/sglang-v7-baseline.yaml` | 当前是 **005**，含有失败的旧 region 回收前置步骤，不能直接重复提交 |
| `research/jax-stack/{PLAN.md,README.md,coverage-review.md,status.json}` | 新范围和队列更新，未提交 |

每次已提交的 YAML 和 producer 快照在 `artifacts/jax-stack/sglang-v7-baseline-NNN/`；
复查某次失败必须读取该次 `submitted.yaml` / `producer.py`，不要用现在已改变的活动脚本替代。
preflight-001 以 submitted.yaml 为生产脚本权威，早期单独抽出的 producer.py 带 YAML 缩进，不是直接可执行副本。

本轮已运行：

- `tools/selftest-project-status.py`：positive summary + 20 negative contracts 通过。
- 新 Python 脚本 AST 语法检查；部分 embedded shell 做了 `bash -n`。
- 变更文档本地链接检查、input ID 唯一性/无重复 pending input、原 index SHA 检查通过。
- `git diff --check -- research/jax-stack deploy/falcon` 通过。
- preflight raw protobuf + Pallas NumPy 输出独立验证通过，ProfileData 事件总数复核通过。
- loaded native/package hash 对应 21 项通过。

**没有完成**：全模型 repeatability、真实 fusion/split、真实峰值下降、device named region 数量/嵌套/开销、LLO 识别、
固定源码 wheel 在 TPU 上的加载、TPU compiler-pass Hack 的三态验收。

## 11. 建议接手执行顺序

1. 先读取交接 snapshot，确认最新 005 failed、lease `[]`。如果时间已过去，重新只读查询；不要直接重提当前 YAML。
2. 决定业务入口：继续当前外部兼容驱动，或切换到正常服务/worker。保持业务文件/权重不变，固定模型 revision。
   去掉会阻断业务的“回收旧 enabled 目录”前置步骤；第三处 forward 返回值适配尚需真机验证。
3. 独立完成设备事件采集的回收闭环：PyPI jax/jaxlib 0.11.1 + libtpu 0.0.46
   （最新已发布组合），default/enabled × plain/named；
   当场取回文件并核对 hashes/exit-code，不以日志成功替代文件验收。运行 raw verifier 和 negative tests。
4. 保存真实基线输入、logits/IDs、trace 和 HLO；检查完整 dtype/mesh/KV 参数一致性。
5. 从真实 MLP 编译图定位已有 fusion 和临时内存，再编写可逆的编译/lowering 注入；先小规模精度门，再长输入 memory/time。
6. 每个独立里程碑更新 PLAN/status/结果和 fingerprints，完成相应语义验证与 selftests。
7. 按上文 Git 隔离发布流程完成 publication-021，严格保留用户原暂存区。不强推，不把忽略目录和私人文档全文上传。

复查最后实验：

```bash
/home/askr/.falcon/bin/falcon exp get exp-uvj2yptpe2 --output json
/home/askr/.falcon/bin/falcon exp logs exp-uvj2yptpe2 --tail 40
/home/askr/.falcon/bin/falcon lease list --job-id job-pp9r06b00d --output json
```

交接时已确定的信息足够继续，不需要再向用户重复询问模型、修改范围和 TPU 访问方式。
若新增问题确实需要用户决策，应提出具体、已调查的问题，而不是重复旧三项输入清单。
