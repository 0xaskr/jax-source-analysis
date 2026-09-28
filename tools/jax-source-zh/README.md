# JAX 源码中文注释处理脚本

这里存放 `basearray`、`core`、`partial-eval` 和 `util` 的处理脚本。对应的原文件、中文工作副本、分块、manifest 和拼接输出仍保存在仓库根目录的 `artifacts/<组件>-py-zh/`。

每个组件的 `apply_fixes.py` 对工作副本应用精确匹配的审阅修订。`core`、`partial-eval` 和 `util` 还提供 `make_chunks.py`、`assemble.py`、`verify.py`：它们分别重新生成分块、从 `out/` 拼接工作副本、只读检查语法与非注释 token。`basearray` 没有专用验证器，可用相同的 `core/verify.py` 检查。

例如，仅做读取检查：

```bash
python3 -B tools/jax-source-zh/util/verify.py \
  artifacts/util-py-zh/util.py.orig artifacts/util-py-zh/util.py
```

`make_chunks.py`、`assemble.py` 和 `apply_fixes.py` 会写入对应的 `artifacts/` 目录。当前 `core-py-zh/` 没有拼接所需的 `manifest.json` 与 `out/`；运行写入脚本前先检查该组件的工作副本与分块状态。
