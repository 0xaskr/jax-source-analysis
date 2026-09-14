# Bazel 外部依赖与 Python 工具链记录

终态构建后，已保存 **199 个缓存 repository 的 451 份顶层描述文件**，并核对 Python
工具链的选择、归档和实际文件。467 个 capture 产物通过验证。结果见
[external-descriptor-results.json](external-descriptor-results.json)；这份记录为 `SOURCE-ONLY`，
描述缓存和构建配置，没有声明完整 action 输入闭包或离线重建成功。

## Python 的两个角色

| 角色 | 本次版本 | 依据 |
|---|---|---|
| 构建驱动及固定镜像基础 venv | 3.12.3 | 固定镜像预检、构建 manifest 与实际运行基线 |
| Bazel 请求的 Python minor | 3.12 | 保存的 generated.jax_configure.bazelrc 与 py_version.bzl |
| Bazel 生成工具链的完整版本 | 3.12.13 | `pythons_hub/versions.bzl` 的 minor mapping 及生成 BUILD.bazel |
| 当前缓存中的该解释器 | 3.12.13 | 本机执行该解释器的版本/ABI 输出，以及归档成员字节核对 |

本次缓存中的 `rules_python` 声明版本为 2.2.0。其 runtime manifest 对应归档是
`20260414/cpython-3.12.13+20260414-x86_64-unknown-linux-gnu-install_only.tar.gz`，
111,512,478 bytes，SHA-256：
`cdcf8724d46e4857f8db5ee9f4252dc2f5da34f7940294ec6b312389dd3f41e0`。

已完整校验归档 hash，并逐字节对应 `bin/python3.12`、`lib/libpython3.12.so.1.0` 和
`include/python3.12/patchlevel.h`。本机版本探测使用 `-I -S -B`，不加载 site 包或写
bytecode。该探测是构建后的环境取证，没有冒充构建期间逐 action 的 Python 进程采样。
公开下载位置及各文件指纹在结果 JSON 中保留。

## 缓存记录的使用范围

每个 repository 记录顶层 MODULE/REPO/WORKSPACE/BUILD 的快照、hash、声明的模块
名称/版本，以及构建后宿主机解析的 symlink 目标。它们适合追查配置与来源；实际编译
使用的源码挂载还要结合保存的 Docker argv，不能由宿主机路径单独还原。

这次 `googletest+` 仍链接到此前 C++ 测试的独立补丁副本，而最终生产 wheel argv
没有 `--override_module=googletest=...`。这个具体观察说明缓存能保留别的配置留下的
条目。所有 `used_by_build` 字段保持 null；要证明某条目被某个 action 消费，仍需对应
目标的依赖/执行记录。此处也不据该 symlink 推断生产 wheel 是否链接了 Googletest。

三份核心归档、43 个 XLA 补丁目标及 1,093 个下载缓存 payload 的核对见
[源码构建说明](source-build.md)。两份审计共同补充构建来源，仍未覆盖所有 action 输入；
当前构建使用 `--lockfile_mode=off`。

## 重放与验证

```bash
.venv/bin/python -B research/jax-stack/audit_external_descriptors.py \
  --output artifacts/jax-stack/external-descriptors-new
.venv/bin/python -B research/jax-stack/verify_external_descriptors.py \
  --capture artifacts/jax-stack/external-descriptors-new --selftest
```

验证器检查完整 capture 清单、描述文件、工具链选择、归档及三个成员的实际字节。
反例覆盖把缓存误报为 action 闭包/目标依赖、混淆 Python 版本、把本机探测误报成
构建期进程采样、遗漏产物，以及首轮缺少 producer provenance 的真实记录。

当前主结果来自 `artifacts/jax-stack/external-descriptors-002`；首轮 001 保留，并继续
被完整 provenance 门槛拒绝。这份审计没有执行 JAX 工作负载或任何 TPU 程序。
