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

## 证据标签

标签描述实际完成的验证，不能按预期运行环境填写。

| 标签 | 含义 | 不能据此声称 |
|---|---|---|
| `SOURCE-ONLY` | 在锁定 revision 上检查了源码、构建定义或 ABI | 代码已编译或行为已发生 |
| `RUN-CPU` | 探针在真实 CPU backend 上成功执行 | TPU lowering、数值或性能等价 |
| `SIM-TPU` | 在 CPU 上使用 TPU/Pallas interpreter 成功模拟 | 真实 TPU timing、资源占用或完整 compiler lowering |
| `COMPILE-TPU` | 针对记录的 TPU topology 编译成功，可以是 compile-only | executable 已在 TPU 上执行 |
| `RUN-TPU` | executable 在记录的真实 TPU 上运行并完成断言 | 其他 TPU 代际或拓扑具有相同行为 |
| `REPLAY-OFFLINE` | 使用锁定工具重放或分析了已有 capture | 原始 capture 在本机重新生成 |

一项结论可以关联多条证据。例如源码入口使用 `SOURCE-ONLY`，CPU 语义使用 `RUN-CPU`，LLO 只使用一次单独的 `RUN-TPU` capture。不得把多个较弱标签合并成更强标签。

## 版本与来源

1. 所有路径均相对仓库根目录，禁止写用户名、主机绝对路径或临时目录。
2. 源码组件必须记录完整 40 位 Git revision 和仓库内 root。包版本不能代替源码 revision。
3. wheel、`libtpu.so` 或其他二进制必须记录 SHA-256；有 build id 时一并记录。
4. capture 必须记录分析仓库 revision、涉及的组件 revision、生成命令、工作目录、backend、架构和设备数量。TPU 证据还要记录设备型号和 topology。
5. `dirty` 默认描述整个组件 root；设置 `scope_paths` 后只描述这些仓库相对路径，便于隔离无关工作区改动。`dirty: true` 时必须列出能重建改动的 patch 及其 SHA-256。没有 patch 的 dirty scope 不能产生可提交的验证证据。
6. 环境只记录必需环境变量的名称。令牌、凭据、项目名、主机名和内部路径不得进入 manifest 或产物。
7. 升级任一源码或二进制后，旧证据保留原 revision；重新生成到新的 capture 目录，不能原地伪装成新版本证据。

同一文件内的 component id、source entry id、evidence id 和 artifact id 必须唯一；所有跨文件引用必须能解析。JSON Schema 不检查这些关系，校验工具或 review 必须检查。

## 生成产物

- 生成产物不可手工编辑。修改 probe、pass 或参数后重新生成，并更新 manifest 中的 SHA-256。
- 原始产物和为了 diff 而规范化的产物分开保存；规范化产物通过 `normalized_from` 指向原始产物，并在 `normalization` 中写清变换。
- IR、proto 或 profile 等二进制产物应附带可 review 的文本表示。默认只提交确定性、可复现且小于 1 MiB 的产物；更大的文件提交最小复现片段，或记录外部位置、SHA-256 和获取方式。
- 去除随机临时路径、PID、墙钟时间等无语义噪声是允许的，但不能删改 op、shape、layout、sharding、pass 名称或错误信息。原始文件仍应保留，除非包含敏感信息。
- 失败结果只在它是预期失败测试或故障样本时提交，并使用 `expected-failure` 结果及明确断言。
- 时间戳用于来源追踪，不作为 golden test 的比较内容。

## 最小示例与校验

`docs/contributing/examples/03-jit-cache/` 展示一个最小主题：它把 `_trace_for_jit` 源码入口与三次 CPU 调用的 tracing 计数关联起来。

从仓库根目录校验 schema 和示例：

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
