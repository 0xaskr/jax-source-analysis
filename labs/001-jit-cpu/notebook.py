import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="JAX JIT CPU Lab")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    from lab_core import cache_experiment
    from lab_core import environment_info
    from lab_core import jaxpr_text
    from lab_core import source_excerpt
    from lab_core import source_rows
    from lab_core import stablehlo_text

    return (
        Path,
        cache_experiment,
        environment_info,
        jaxpr_text,
        source_excerpt,
        source_rows,
        stablehlo_text,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Lab 001 · `jax.jit` 的 CPU 执行路径

    **这不是静态文章。** 调整参数会立即重新 trace 函数并生成新的 Jaxpr 和
    StableHLO；点击缓存实验可以看到相同 shape 与新 shape 的行为差异。
    """)
    return


@app.cell(hide_code=True)
def _(environment_info, mo):
    _environment = environment_info()
    mo.hstack(
        [
            mo.stat(_environment["JAX"], label="JAX 版本", bordered=True),
            mo.stat(_environment["CPU device"], label="执行设备", bordered=True),
            mo.stat("editable", label="源码加载方式", bordered=True),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    size = mo.ui.slider(
        1,
        16,
        value=4,
        step=1,
        show_value=True,
        label="输入长度（shape）",
        full_width=True,
    )
    scale = mo.ui.slider(
        -4.0,
        4.0,
        value=2.0,
        step=0.5,
        show_value=True,
        label="scale",
        full_width=True,
    )
    bias = mo.ui.slider(
        -4.0,
        4.0,
        value=1.0,
        step=0.5,
        show_value=True,
        label="bias",
        full_width=True,
    )
    debug_info = mo.ui.checkbox(value=False, label="在 StableHLO 中保留调试位置")
    mo.vstack(
        [
            mo.md("## 1. 改变实验参数"),
            mo.hstack([size, scale, bias], widths="equal", gap=1),
            debug_info,
        ],
        gap=1,
    )
    return bias, debug_info, scale, size


@app.cell(hide_code=True)
def _(bias, mo, scale, size):
    mo.callout(
        mo.md(
            f"""
            当前实验函数：

            ```python
            def affine(x):
              return x * {scale.value:g} + {bias.value:g}
            ```

            输入抽象值：`float32[{size.value}]`
            """
        ),
        kind="info",
        title="当前程序",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## 2. 沿编译路径观察"),
            mo.mermaid(
                """
                flowchart LR
                  A[Python] --> B[jax.jit]
                  B --> C[Tracing]
                  C --> D[Jaxpr]
                  D --> E[MLIR lowering]
                  E --> F[StableHLO]
                  F --> G[XLA compile]
                  G --> H[PJRT CPU]
                  H --> I[Result]
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(bias, debug_info, jaxpr_text, mo, scale, size, stablehlo_text):
    _jaxpr = jaxpr_text(size.value, scale.value, bias.value)
    _stablehlo = stablehlo_text(
        size.value,
        scale.value,
        bias.value,
        debug_info=debug_info.value,
    )
    mo.ui.tabs(
        {
            "Jaxpr": mo.md("```text\n" + _jaxpr + "\n```"),
            "StableHLO": mo.md("```mlir\n" + _stablehlo + "\n```"),
        },
        value="Jaxpr",
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo, source_rows):
    _rows = source_rows()
    source_picker = mo.ui.dropdown(
        [row["Symbol"] for row in _rows],
        value="jax.jit",
        label="选择一个 symbol 查看当前本地源码",
        searchable=True,
        full_width=True,
    )
    mo.vstack(
        [
            mo.md("## 3. 把阶段映射回当前源码"),
            mo.ui.table(
                _rows,
                pagination=False,
                selection=None,
                show_column_summaries=False,
                show_data_types=False,
                show_download=False,
                show_search=True,
            ),
            source_picker,
        ],
        gap=1,
    )
    return (source_picker,)


@app.cell(hide_code=True)
def _(mo, source_excerpt, source_picker):
    _location, _source = source_excerpt(source_picker.value)
    mo.callout(
        mo.md(f"`{_location}`\n\n```python\n{_source}\n```"),
        kind="neutral",
        title=source_picker.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    run_cache = mo.ui.run_button(
        label="运行三次并观察 tracing cache",
        kind="success",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md("## 4. 用真实执行验证缓存结论"),
            mo.md("依次运行：相同 shape、相同 shape、`shape + 1`。"),
            run_cache,
        ]
    )
    return (run_cache,)


@app.cell(hide_code=True)
def _(bias, cache_experiment, mo, run_cache, scale, size):
    mo.stop(
        not run_cache.value,
        mo.callout("点击上方按钮后才会触发编译和执行。", kind="neutral"),
    )
    _observations = cache_experiment(size.value, scale.value, bias.value)
    _rows = [observation.as_row() for observation in _observations]
    mo.vstack(
        [
            mo.ui.table(
                _rows,
                pagination=False,
                selection=None,
                show_column_summaries=False,
                show_data_types=False,
                show_download=False,
                show_search=False,
            ),
            mo.callout(
                "第二次调用的 tracing 次数不变；第三次改变 shape 后增加一次。",
                kind="success",
                title="观察结果",
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(Path, mo):
    _patch_path = Path(__file__).resolve().parent / "patches" / "trace-pjit.patch"
    _patch = _patch_path.read_text(encoding="utf-8")
    mo.vstack(
        [
            mo.md("## 5. 连接真实 Hack"),
            mo.accordion(
                {
                    "查看 `_trace_for_jit` 补丁": mo.md(
                        "```diff\n" + _patch + "\n```"
                    ),
                    "应用、运行与撤销": mo.md(
                        """
                        ```bash
                        git -C upstream/jax apply ../../labs/001-jit-cpu/patches/trace-pjit.patch
                        JAX_SOURCE_ANALYSIS_TRACE_JIT=counted_affine \\
                          uv run python labs/001-jit-cpu/probe.py --stage run
                        git -C upstream/jax apply -R ../../labs/001-jit-cpu/patches/trace-pjit.patch
                        ```

                        这是 Python-level Hack，editable 安装会立即读取修改后的源码，
                        因此本 Lab 不需要手动编译。
                        """
                    ),
                },
                multiple=False,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        "继续向 StableHLO 后端、XLA CPU codegen 和 LLVM 深入时，再拆分新的 Lab。",
        kind="warn",
        title="当前边界",
    )
    return

if __name__ == "__main__":
    app.run()
