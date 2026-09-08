# 从固定源码构建 CPU jaxlib

## 目的

当前环境把 `upstream/jax` 作为 editable Python package 使用，但加载的 C++ runtime 是 PyPI `jaxlib==0.11.1`。JAX 源码版本是 `0.11.2.dev20260830+5832e86644`，所以当前环境适合研究 Python transformation，不能作为 `upstream/xla` C++ 源码已经执行的证据。

本构建闭环需要产生一个由以下源码生成的 CPU wheel：

```text
JAX  5832e866449a41c3eea6333416528039119a0fde
XLA  496bd4bd49db9ecbffd85da630b49c860b724604
Python 3.12
```

构建时必须传入 `--local_xla_path`。这样后续对 `upstream/xla` 的修改才会进入 wheel，而不是由 Bazel 重新取得另一个 external repository。

JAX 固定版本的原始构建说明位于 [`upstream/jax/docs/developer.md`](../../upstream/jax/docs/developer.md)，构建入口是 [`upstream/jax/build/build.py`](../../upstream/jax/build/build.py)。本页只记录本分析仓库的额外 provenance、安装和回滚约束。

## 当前前置检查

从仓库根目录运行：

```bash
.venv/bin/python tools/check-jaxlib-build-env.py
```

严格模式只接受本仓库记录的 Linux x86_64 本地工具和输入边界：Python 3.12.3、固定 SHA-256 的 Bazel 8.7.0、`/usr/bin/clang{,++}` 18.1.3、SHA-256 固定的 `/usr/bin/git` 2.43.0，以及处于固定 commit 且 tracked/untracked 状态干净的 JAX/XLA 源码树。Git 检查使用固定绝对路径、空白全局/系统配置、关闭的 fsmonitor 和 hooks，不接受调用 shell 的 `PATH` 或 `GIT_*`。任一版本、命令返回码、hash、commit 或 dirty state 不符都会返回非零：

```bash
.venv/bin/python tools/check-jaxlib-build-env.py --strict
```

2026-09-08 的当前检查结果：

- Python 3.12.3 可用；
- JAX 要求 Bazel 8.7.0；
- JAX `build.py` 已下载并校验固定的 Bazel 8.7.0；二进制位于 `upstream/jax/bazel-8.7.0-linux-x86_64`，由上游工作树忽略；
- Clang 18.1.3 已安装，固定入口为 `/usr/bin/clang` 和 `/usr/bin/clang++`；
- 工作区约有 929 GiB 可用空间；
- 主机约有 15 GiB RAM、4 GiB swap 和 12 个逻辑 CPU。

正式构建前重新运行 preflight。下面的构建包装器会把校验结果和输入 fingerprint 在启动子进程之前原子写入 manifest。preflight 校验当前 `sys.executable` 正是仓库固定的 `.venv/bin/python`，其 base executable 固定为 `/usr/bin/python3.12` 并匹配记录的大小和 SHA-256。记录和执行的 argv 从 `upstream/jax` 使用等价的 `../../.venv/bin/python -S`，禁止启动时加载 venv 的 `.pth` 或 `sitecustomize`。

## 配置检查

正式长时间构建前，先让仓库包装器执行 JAX 的 `--dry_run`，输出将要执行的 Bazel 命令：

```bash
.venv/bin/python tools/build-jaxlib.py \
  --build-id=cpu-baseline-config-20260908 \
  --dry-run
```

该命令固定使用 Bazel 8.7.0、`/usr/bin/clang` 和 `--jobs=4`，并把 manifest 写到 `manifests/build-fingerprints/cpu-baseline-config-20260908.json`。第一次 attempt 的日志与空 wheel 目录分别位于 `artifacts/builds/cpu-baseline-config-20260908/attempts/0001/build.log` 和 `attempts/0001/wheels/`。dry-run 成功只记录为 `dry-run-succeeded`，不会记录为 wheel 构建成功。

JAX `build.py` 即使在 `--dry_run` 下也会覆盖 ignored 文件 `upstream/jax/.jax_configure.bazelrc`。包装器会把它复制到本 attempt 的 artifact 目录并记录副本的大小和 SHA-256；检查它时不要把 dry-run 当成只读操作。

manifest root 只能是 `manifests/build-fingerprints/` 或其子目录，artifact root 只能是 `artifacts/builds/` 或其子目录；从仓库根开始的任一现存路径分量均不得是符号链接。JAX/XLA source root 同样必须是仓库内的真实目录，且 Git top-level 必须精确等于各自固定根。构建会清除可能影响编译闭包的 `CC`、`CXX`、`CFLAGS`、`LDFLAGS`、`BAZEL_*`、`HERMETIC_*`、`ML_WHEEL_*`、`TF_*`、`PYTHONPATH` 和 `XLA_FLAGS` 等继承变量，并固定 `PYTHONNOUSERSITE=1` 与 `PYTHONDONTWRITEBYTECODE=1`。源码树中预存的 `.pyc`/`.pyo` 会被拒绝，构建子进程也不会生成新的字节码。Bazel 总是使用 `--nosystem_rc` 和 `--nohome_rc`，仍会读取仓库自身的配置。

需要确认输出至少满足：

1. wheel target 是 `//jaxlib/tools:jaxlib_wheel`；
2. hermetic Python 是 3.12；
3. XLA override 指向本仓库的 `upstream/xla`；
4. compiler 配置包含 `clang_local`、`CC=/usr/bin/clang` 和 `CXX=/usr/bin/clang++`；
5. Bazel 参数包含 `--jobs=4`，且没有启用 CUDA、ROCm 或 OneAPI plugin；
6. 输出路径位于 `artifacts/builds/<build-id>/attempts/0001/wheels`，不覆盖当前 `.venv`。

## 首个基线构建

正式构建使用一个新的稳定 build id：

```bash
.venv/bin/python tools/build-jaxlib.py \
  --build-id=cpu-baseline-20260908
```

包装器实际在 `upstream/jax` 中执行等价于以下参数的命令，并把 argv 原样写入 fingerprint：

```text
../../.venv/bin/python -S build/build.py build
--wheels=jaxlib
--python_version=3.12
--local_xla_path=../xla
--output_path=../../artifacts/builds/cpu-baseline-20260908/attempts/0001/wheels
--bazel_path=./bazel-8.7.0-linux-x86_64
--clang_path=/usr/bin/clang
--bazel_startup_options=--nosystem_rc
--bazel_startup_options=--nohome_rc
--bazel_options=--jobs=4
--bazel_options=--lockfile_mode=off
--verbose
```

15 GiB 内存下先固定使用四个 Bazel jobs。若仍受内存限制，通过重复的 `--bazel-option` 把额外资源限制交给包装器；这些参数会进入 argv 和输入 fingerprint。包装器只接受不会改变目标、源码、工具链或生成代码语义的资源/诊断参数：`--local_resources=...`、`--ram_utilization_factor=...`、`--local_cpu_resources=...`、`--local_ram_resources=...`、`--experimental_ui_max_stdouterr_bytes=...`，以及 `--announce_rc`、`--sandbox_debug`、`--show_timestamps`、`--subcommands`。例如 `--override_repository`、`--config` 和 `--repo_env` 会被拒绝。

需要显式选择 Bazel output user root 时使用重复的 `--bazel-startup-option`。startup 扩展只允许 `--output_user_root`、`--max_idle_secs`、`--host_jvm_args=-Xms/-Xmx` 和 batch 开关；output user root 必须位于 `/tmp` 或本仓库 `artifacts/builds` 下。例如首轮
探索构建已经在 `/tmp/jax-source-analysis-bazel` 填充 cache，正式留证运行可以传
`--bazel-startup-option=--output_user_root=/tmp/jax-source-analysis-bazel` 复用它；该
路径和参数会进入 fingerprint。`/tmp` cache 可丢弃，wheel、manifest 和日志仍写入
上述持久目录。恢复同一 build id 时必须传入完全相同的 startup/build options。

子进程把 stdout/stderr 直接追加到当前 attempt 的 durable `build.log`，终端只显示 build id、attempt、日志位置与最终状态。日志不经过 wrapper 的 pipe，因此 wrapper 被强制终止后，存活的构建子进程仍可继续写日志。不使用上游 `--detailed_timestamped_log`：该实现会把完整输出累积在内存中，但不会在本仓库保存独立日志文件。

需要观察进度时，在另一个终端对状态命令输出的日志路径运行 `tail -f`。

## 状态与恢复

随时可以只读查看状态：

```bash
.venv/bin/python tools/build-jaxlib.py \
  --build-id=cpu-baseline-20260908 \
  --status
```

状态为 `build-failed`、`interrupted`，或记录为 `running` 但 wrapper/child identity 均已失效时，可以用相同输入恢复：

```bash
.venv/bin/python tools/build-jaxlib.py \
  --build-id=cpu-baseline-20260908 \
  --resume
```

每个 build id 有独占文件锁；另有全局锁保护共享的 `.jax_configure.bazelrc`。锁文件通过已验证的仓库目录描述符以 `O_NOFOLLOW` 打开，并要求是单链接普通文件。锁描述符会传给构建子进程，所以 wrapper 意外退出后，只要子进程仍在运行，新进程就不能并发恢复。manifest 使用 Linux boot ID、`/proc/<pid>/stat` 的 start time 和 cmdline SHA-256 共同识别进程，不会把重用的 PID 当成原构建。

子进程结束并释放锁后，恢复会把失效的 `running` attempt 标为 `interrupted`，并在新的 `attempts/<NNNN>/` 下安全重跑。每个 attempt 使用独立的 `HOME`，因此默认 Bazel cache 不跨 attempt 复用；只有显式传入完全相同且安全的 `--output_user_root` 才会复用。旧 attempt 的 wheel 永远不会作为新 attempt 的产物。源码、工具 hash、构建配置或 build id 变化时包装器拒绝恢复，要求新建 build id。记录仍有存活 identity，或已经是 `build-succeeded`/`dry-run-succeeded` 时也拒绝重复运行。正式命令只有在子进程返回零、生成 `.jax_configure.bazelrc`，且当前 attempt 目录中恰有一个新的 `jaxlib-*.whl` 后才写 `build-succeeded`。

## Wheel fingerprint

包装器在构建启动前记录输入，在每次 attempt 启动和结束时原子更新状态。读取已有 manifest 时会先执行 Draft 2020-12 schema，重放固定 source/tool/input/patch/path 条件，并检查 attempt、result 和时间顺序。构建 manifest 至少包含：

```text
JAX/XLA commit 和 dirty state
关键 lock/config/build.py 文件的 SHA-256
被 preflight 检查且实际执行 build.py 的同一个 Python
Bazel version
C/C++ compiler path and version
all Bazel startup/build options
每个 attempt 独立的 command、log、wheel directory
本次 wheel filename、size 和 SHA-256
各 attempt 的 boot ID/start time/cmdline identity、start/end time、exit code 和 result
```

fingerprint 写入 `manifests/build-fingerprints/`，结构由 `manifests/schema/build-jaxlib.schema.json` 约束。完成状态的 manifest 可以 review 后提交。`artifacts/builds/` 已整体忽略，其中的 wheel、完整日志和其他大型文件不提交到 Git；manifest 只保存其仓库相对位置、大小和 SHA-256。

`build-succeeded` 只表示固定构建命令成功，并生成一个结构有效、RECORD 哈希完整的 `jaxlib-*.whl`。wrapper 会把 `runtime_validation.status` 强制保持为 `not-run`；当前 schema 也拒绝手工写入 `passed` 或 `failed`。它不等同于 `RUN-CPU` 证据，也不证明 wheel 可以导入或包含预期的本地 XLA 改动。

## 隔离安装与验证

不要直接覆盖项目当前 `.venv`。首个 source-built wheel 使用临时 uv environment 或独立 venv 验证。验证程序至少输出并断言：

- `jax.__file__` 仍指向固定的 editable `upstream/jax`；
- `jaxlib.__file__` 指向临时环境中新安装的 wheel；
- JAX/jaxlib compatibility check 通过；
- 仅发现预期 CPU devices；
- Lab 001 的 Jaxpr、StableHLO、数值和 cache probe 通过；
- 一个直接触达新增 C++ 诊断标记的 probe 通过。

隔离验证 capture 还必须记录 jaxlib 版本、embedded git hash、native library 路径、动态依赖，以及所加载 wheel 的 SHA-256。P1 将先定义一个不会与 build manifest validator 形成循环依赖的哈希化运行证据合同；在该合同落地前，运行结果只能进入独立 capture，不能修改这里的 `runtime_validation` 占位字段。

## 证明本地 XLA 修改进入运行时

基线 wheel 成功后增加一个默认关闭、环境变量控制的无语义诊断点。诊断点应位于后续 pass 实验会经过的 XLA 路径，并具备以下性质：

1. 未设置环境变量时没有输出或行为变化；
2. patch 可以用 `git apply --check` 检查并反向应用；
3. 重新构建后，probe 能观察唯一版本字符串或诊断事件；
4. 使用原始 PyPI wheel 时 probe 不会误报；
5. 移除 patch 并重建后恢复基线。

这个步骤证明“阅读的 XLA 工作树”“构建输入”和“实际加载的二进制”是同一个版本闭包。

## 安装、缓存与回滚

每次切换 wheel 时同时处理以下状态：

- Python environment 中的 jaxlib package；
- JAX in-memory compilation cache；
- persistent compilation cache；
- Bazel build cache；
- 已注册的 HLO transformations；
- 当前 shell 中的 `JAX_*`/`XLA_FLAGS` 配置。

回滚完成的判据是：

1. `jaxlib` 路径、版本和 hash 回到记录的基线；
2. JAX/JAXLIB import 与 Lab 001 通过；
3. JAX 和 XLA submodule 的工作树状态符合 fingerprint；
4. 诊断标记不再出现。

## 尚未执行

本文目前是构建设计，证据等级为 `SOURCE-ONLY`。Clang 18.1.3 和固定 Bazel 8.7.0 已准备完成，source-built wheel 尚未生成。完成首次构建后，应把真实命令、耗时、资源设置、wheel fingerprint 和验证结果替换进本页，并更新 `PLAN.md` 的 P1 状态。

## 构建闭包补充约束

正式 wrapper 拒绝 `upstream/jax/.bazelrc.user` 的任何目录项，包括普通文件、目录、正常或悬空符号链接。它把 JAX `.bazelrc`、`.bazelrc.user` 和 `MODULE.bazel.lock` 的存在状态及普通文件字节纳入 fingerprint，并在启动前与子进程结束后重新采集。这个 revision 没有 tracked `MODULE.bazel.lock`，且 JAX 会忽略生成的 lock，因此命令固定加入 `--bazel_options=--lockfile_mode=off`：Bazel 不读取也不改写可能存在的 ignored lock。

子进程从空环境建立固定的 `PATH`、`C.UTF-8` locale、`TZ=UTC`、`PYTHONNOUSERSITE=1` 与 `PYTHONDONTWRITEBYTECODE=1`，并使用当前 attempt 下的 `home/` 与 `tmp/`。只按名称继承 allowlist 中实际存在的 proxy/证书变量；manifest 当前不保存这些变量的值或证书文件/目录的内容身份。因此使用任一继承变量的构建也不能称为完整可重放闭包；P1 需要补充值哈希和证书内容清单，或禁用继承。Clang 与 Clang++ 的实际二进制 SHA-256 进入 toolchain fingerprint。

默认仍要求 JAX/XLA 工作树干净。受控源码构建先保存规范 patch：

```bash
git -C upstream/xla diff --binary --full-index --no-color \
  --no-ext-diff --no-textconv HEAD -- . > patches/my-xla-change.patch
.venv/bin/python tools/build-jaxlib.py \
  --build-id=xla-change-20260908 \
  --source-patch=xla=patches/my-xla-change.patch
```

`--source-patch` 仅支持 `jax`/`xla`。wrapper 要求 HEAD 是 fixed base，拒绝 untracked 文件、冲突项、changed/dirty nested submodule、assume-unchanged/skip-worktree index 标记，以及任何预存的 ignored `.pyc`/`.pyo`；patch 字节必须精确等于上述 diff，并能用 temporary Git index 从 fixed base 正向 `git apply --check --cached`。ignored 文件按 fail-closed allowlist 检查，只允许固定并单独哈希的 Bazel binary/config 输入、明确的 Bazel 生成输出和 editable-install egg-info。验证通过只豁免对应的 `source-dirty:<component>` blocker。component、base revision、patch 路径、大小和 SHA-256 都被记录并在构建前后复核。

构建前后的不可变输入 fingerprint 会忽略 `.jax_configure.bazelrc` 和标准 Bazel output symlink 从 absent 到 present 的受控变化，因为它们是本次命令的输出；任何其他新增 ignored 路径仍会改变 fingerprint 或触发 blocker。历史 manifest 的只读验证会从 pinned commit 和保存的 patch bytes 在 temporary Git index 中重放可应用性，因此源码回滚为 clean 后仍可验证。启动和恢复路径额外要求 live tree 的完整 diff 与 patch bytes 精确相等。

这仍是在 live source tree 上构建。开始/结束快照会发现稳定漂移，但无法消除检查与实际文件读取间的竞态；更强 provenance 需要未来从 fixed commit 和已验证 patch 建立隔离 source tree。

每个 attempt 把本次生成的 `.jax_configure.bazelrc` 归档到固定的 `generated.jax_configure.bazelrc`，并在关闭 log 后记录 log 的路径、大小和 SHA-256。wheel 必须直接位于本 attempt 的 `wheels/`，三处 project/version（filename、`.dist-info`、`METADATA`）必须规范化后完全一致且等于 pinned `jaxlib 0.11.2.dev0+selfbuilt`；filename 与 `WHEEL` tag 必须都是 `cp312-cp312-manylinux_2_27_x86_64`，并至少包含 `_jax.so`、`cpu_feature_guard.so` 和 `libjax_common.so`。RECORD 必须覆盖 ZIP 中每个文件并验证 SHA-256/大小。`--status`、`--resume` 和 terminal 状态写入会复核现存 wheel、log 与归档 Bazel rc 的真实字节；`--status` 输出最后一个 log。manifest 内的仓库所有路径使用仓库相对形式，不记录 home、用户名或临时 fixture 根。历史验证会从 `inputs.analysis_repository_commit` 重建 wrapper/schema/config/lock 文件，并从 pinned JAX commit 加保存的 patch 重建 JAX 输入，逐项核对记录的存在状态、大小和 SHA-256；resume 还会核对当前工作树字节。

当前 manifest 还没有固定完整的外部 Bazel 模块闭包。命令使用 `--lockfile_mode=off`，只把 XLA 切换为 local override；registry、module extension 和 resolved repository 状态尚未归档。因此现阶段不能把 `build-succeeded` 描述成可从 JAX/XLA commit 单独重现的完整源码闭包。正式 P1 构建前需要从 clean isolated source 生成并审查 lock，归档并哈希 `bazel mod graph`、resolved repositories 和下载完整性，再用固定的 error/update 策略重放。

可以只读验证任意默认 manifest 子树中的记录：

```bash
.venv/bin/python tools/build-jaxlib.py \
  --verify-manifest manifests/build-fingerprints/<build-id>.json
```

Python 调用方也可使用 `load_and_validate_manifest(path)`；公共入口始终验证所记录 artifact 的真实字节，没有 structural-only 开关。正式成功值是 `status == "build-succeeded"`；源码 revision 位于 `inputs.sources.jax.commit` 和 `inputs.sources.xla.commit`，wheel identity 位于 `result.wheels[]`。持久合同 selftest 不调用真实 build：

```bash
.venv/bin/python tools/selftest-build-jaxlib.py
```

`build-succeeded` 只证明构建与产物闭包；runtime validation 在后续哈希化 capture 合同完成前保持 fail-closed。
