# 源码分析证据约定

本约定用于让源码导读中的结论可以定位、复现和复查。每个主题必须说明它分析的版本、源码入口、运行或静态证据，以及证据不能证明什么。

## 主题证据包

一个主题放在 `docs/topics/<topic-id>/`，其中 `<topic-id>` 使用 `NN-kebab-case`。最低要求如下：

```text
docs/topics/<topic-id>/
├── README.md
├── topic.json
├── source-index.json
├── probes/
└── captures/
    └── <capture-id>/
        ├── manifest.json
        └── <generated artifacts>
```

- `README.md` 解释问题、结论、边界和实验，不复制大段上游源码。
- `topic.json` 列出可独立核验的 claim。每条 claim 关联源码条目、产物、复现命令和局限。
- `source-index.json` 把文档概念锚定到精确 revision 下的 symbol。symbol 是主锚点，行号只是该 revision 的阅读提示。
- `probes/` 保存短小的可执行探针；探针失败时应返回非零状态。
- `captures/` 保存值得 code review 或离线回放的生成产物。每次 capture 有自己的 manifest，不能把不同机器、版本或命令的结果混入同一目录。

完整主题可以增加 `callgraph.mmd`、`invariants.md`、`failures.md`、`patches/` 和 `tests/`。这些文件不能替代上述三个 JSON 元数据文件。

对应 schema：

- `manifests/schema/topic.schema.json`
- `manifests/schema/source-index.schema.json`
- `manifests/schema/capture.schema.json`
- `manifests/schema/native-binaries.schema.json`
- `manifests/schema/baseline.schema.json`

## 证据等级与 qualifier

`evidence_level` 描述实际完成的验证，不能按预期运行环境填写。它在下表六级中
只能取一个值；与证据强度正交的状态放入 `qualifiers` 数组。

| 等级 | 含义 | 不能据此声称 |
|---|---|---|
| `SOURCE-ONLY` | 在锁定 revision 上检查了源码、构建定义或 ABI | 代码已编译或行为已发生 |
| `RUN-CPU` | 探针在真实 CPU backend 上成功执行 | TPU lowering、数值或性能等价 |
| `SIM-TPU` | 在 CPU 上使用 TPU/Pallas interpreter 成功模拟 | 真实 TPU timing、资源占用或完整 compiler lowering |
| `COMPILE-TPU` | 针对记录的 TPU topology 编译成功，可以是 compile-only | executable 已在 TPU 上执行 |
| `RUN-TPU` | executable 在记录的真实 TPU 上运行并完成断言 | 其他 TPU 代际或拓扑具有相同行为 |
| `REPLAY-OFFLINE` | 使用锁定工具重放或分析了已有 capture | 原始 capture 在本机重新生成 |

当前定义的 qualifier 只有 `VERSION-SKEW`：它表示运行时二进制与 claim 所引用的
源码 revision 不一致。它不能替代 `evidence_level`，也不能因为已经写入 limitations
而省略。baseline、topic 和 capture 都使用 `evidence_level` 与 `qualifiers` 两个字段。

一项结论可以关联多条证据。例如源码入口使用 `SOURCE-ONLY`，CPU 语义使用
`RUN-CPU`，LLO 编译产物使用单独的 `COMPILE-TPU` capture。不得把多个较弱等级
合并成更强等级。`RUN-CPU`、`SIM-TPU`、`COMPILE-TPU`、`RUN-TPU` 和
`REPLAY-OFFLINE` 必须关联 capture manifest；topic 和 capture 的 level 与 qualifiers
必须一致。

## 版本与来源

1. capture 中的源码、输入和提交产物路径都使用 `/` 分隔、采用规范拼写并相对仓库根目录；
   路径不能越过仓库，也不能通过符号链接逃逸。build manifest 可以记录已脱敏的机器工具链
   绝对路径和 `/tmp` 下的临时 cache，但不能记录 home、用户名或内部主机路径。
2. 源码组件必须记录完整 40 位 Git revision 和仓库内 root。包版本不能代替源码 revision。
3. wheel、`libtpu.so` 或其他二进制必须记录 SHA-256；有 build id 时一并记录。仓库中
   可访问的 binary 使用 `artifact_path`；不能提交的 private binary 使用不含内部主机名
   或路径的 opaque `artifact_uri`、`retrieval` 和 SHA-256。运行证据还要记录实际加载的
   native binaries 指纹，不能只写包版本。
4. capture 必须记录分析仓库 revision、涉及的组件 revision、结构化 `argv`、工作目录、backend、架构和设备数量。`producer` 不保存 shell 字符串；stdout 重定向用 `stdout_path` 表示。TPU 证据还要记录设备型号和 topology。
5. `dirty` 默认描述整个组件 root；设置 `scope_paths` 后只描述这些仓库相对路径，便于隔离无关工作区改动。`dirty: true` 时必须列出能重建改动的 binary-safe patch 及其 SHA-256。patch 必须覆盖 scope 中的 staged、tracked、untracked 和 ignored 改动；没有完整 patch 的 dirty scope 不能产生可提交的验证证据。`dirty: false` 且当前 HEAD 等于记录 revision 时，工作区必须确实干净。
6. 环境只记录必需环境变量的名称。`argv`、版本、retrieval 文本和产物中不得出现控制字符、令牌、凭据、用户名、authority 或内部主机路径。
7. 升级任一源码或二进制后，旧证据保留原 revision；重新生成到新的 capture 目录，不能原地伪装成新版本证据。

同一文件内的 component name/id、source entry id、evidence id 和 artifact id 必须唯一；
所有跨文件引用必须能解析。JSON Schema 不能检查的关系由
`tools/validate-evidence.py` 检查。

`topic.json` 的 `artifact_refs` 保存 capture 中的 artifact id，而不是文件路径。
`source-index.json` 的 `evidence_refs` 与 topic 的 `source_refs` 必须双向对应；topic
中的 `reproduce` 也必须与 capture 的 `producer` 完全相同。`verified` topic 引用的
capture 必须为 `pass`。执行证据通过 `runtime_inventory` 指向 native-binaries
artifact；其中每个 package 都要关联 source component、build revision 和可核验的
wheel/build identity，校验器据此推导是否必须带 `VERSION-SKEW`。package 与对应 binary
component 还要声明相同的 `runtime_roles`：`RUN-CPU` 需要 `cpu-backend`，`RUN-TPU`
需要 `tpu-runtime`，`SIM-TPU` 需要 `tpu-simulator`，`COMPILE-TPU` 需要
`tpu-compiler` 或 `tpu-runtime`。

`lock-artifact` 能把包版本和平台唯一绑定到 lock 中带 hash/size 的候选 wheel，并同时
核验当前加载的 native 文件字节；若未保留 wheel 本体或其他受 wheel hash 绑定的安装
证明，它不能单独证明当前环境中的文件一定由该 wheel 解包而来。需要此强度时，应保留
wheel 作为仓库产物或改用经过完整产物校验的 `build-manifest` identity。

`build-manifest` identity 同时绑定 manifest 自身的 hash/size、build id 和 wheel 的
hash/size。校验器调用构建工具的完整只读校验入口，检查固定 schema、输入指纹、preflight、
连续 attempt、终态、命令、隔离目录和所有现存产物字节；仅仅伪造一个看似成功的 JSON
不能形成 binary provenance。

dirty source evidence 由固定 revision、明确的 `scope_paths` 和完整 patch 重建。capture
的 inputs 必须精确列出重建后 scope 中的所有文件并使用重建内容的 hash；当前 HEAD
恰好等于记录 revision 时，默认验证仍只依赖 revision、patch 和 inputs，使历史 capture
在工作区回滚后保持可重放。采集当时应额外运行
`tools/validate-evidence.py --require-live-source-state`，让当前 scoped worktree 与重建结果
逐字节一致，并拒绝 patch 未覆盖的 tracked、untracked、ignored 或 binary 改动。

`source-index.json` 的 path 必须位于 component root，且必须能从记录 revision 读取为普通
文件；symbol 必须出现在 location 指定的 revision 行窗口。符号链接和 gitlink 不能充当
源码输入。

`REPLAY-OFFLINE` capture 的 backend 固定为 `offline`。它用 `origin_capture`、
`origin_sha256` 和 `origin_artifact_refs` 精确选择已经验证过的原始产物，并记录仓库中的
回放工具 component。校验器递归验证 origin capture、所选 artifact 的 hash 和 replay DAG，
拒绝缺失引用或循环。

## 生成产物

- 生成产物不可手工编辑。修改 probe、pass 或参数后重新生成，并更新 manifest 中的 SHA-256。
- 原始产物和为了 diff 而规范化的产物分开保存；规范化产物通过 `normalized_from` 指向原始产物，并在 `normalization` 中写清变换。
- IR、proto 或 profile 等二进制产物应附带可 review 的文本表示。默认只提交确定性、可复现且小于 1 MiB 的产物；更大的文件提交最小复现片段，或记录外部位置、SHA-256 和获取方式。
- 去除随机临时路径、PID、墙钟时间等无语义噪声是允许的，但不能删改 op、shape、layout、sharding、pass 名称或错误信息。原始文件仍应保留，除非包含敏感信息。
- 失败结果只在它是预期失败测试或故障样本时提交，并使用 `expected-failure` 结果及明确断言。
- 时间戳用于来源追踪，不作为 golden test 的比较内容。

## 最小示例与校验

`docs/contributing/examples/03-jit-cache/` 展示一个最小主题：它把 `_trace_for_jit` 源码入口与三次 CPU 调用的 tracing 计数关联起来。

从仓库根目录完成 schema、引用、revision、路径、唯一性和 SHA-256 校验：

```bash
.venv/bin/python tools/validate-evidence.py
```

对校验器规则本身运行隔离的正例与负例测试：

```bash
.venv/bin/python tools/selftest-validate-evidence.py
```

只做单个 JSON 文件的 schema 调试时，可以使用：

```bash
python3 - <<'PY'
import json
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker

pairs = [
    ("manifests/schema/topic.schema.json",
     "docs/contributing/examples/03-jit-cache/topic.json"),
    ("manifests/schema/source-index.schema.json",
     "docs/contributing/examples/03-jit-cache/source-index.json"),
    ("manifests/schema/capture.schema.json",
     "docs/contributing/examples/03-jit-cache/captures/jit-cold-call/manifest.json"),
]
for schema_path, instance_path in pairs:
  schema = json.loads(Path(schema_path).read_text())
  instance = json.loads(Path(instance_path).read_text())
  Draft202012Validator.check_schema(schema)
  Draft202012Validator(schema, format_checker=FormatChecker()).validate(instance)
  print("OK", instance_path)
PY
```

校验 schema 只能证明结构正确。reviewer 仍需核对 symbol、revision、命令、checksum、claim 和 limitations 是否与真实证据一致。
