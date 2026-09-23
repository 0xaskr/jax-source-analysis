#!/usr/bin/env python3
# Copyright 2026 The JAX Source Analysis Authors.
#
# Generates the SVG diagrams for the JAX repository structure analysis:
#   figures/jax-repo-structure.svg   -- source tree of jax-ml/jax @ v0.11.1
#   figures/jax-repo-pipeline.svg    -- upstream/downstream execution chain
#
# Usage:
#   python3 -B research/software-stack/jax-repo/make_diagrams.py
#
# The generator is deterministic: running it again rewrites both SVGs byte for
# byte. It reads no source files; the tree below is the curated analysis result
# and every node corresponds to a path present in the pinned checkout.

from __future__ import annotations

import pathlib
import html

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "figures"

CJK = ("Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei",
       "WenQuanYi Micro Hei", "sans-serif")
MONO = ("JetBrains Mono", "DejaVu Sans Mono", "Noto Sans Mono CJK SC",
        "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Menlo", "monospace")


def esc(text: str) -> str:
  return html.escape(text, quote=True)


# ---------------------------------------------------------------------------
# Diagram 1: repository structure tree
# ---------------------------------------------------------------------------

# kind -> (band fill, accent, text)
KIND = {
    "root": ("#12263f", "#12263f", "#ffffff"),
    "pkg":  ("#eef4fb", "#2f6fb0", "#12395e"),
    "py":   ("#eefaf3", "#2f9e6b", "#0d4a31"),
    "cpp":  ("#fdf3e7", "#c07a2b", "#6b3f08"),
    "bld":  ("#f4effb", "#7a5cc4", "#3d2a6b"),
    "test": ("#f2f4f7", "#8a94a6", "#39424f"),
    "ext":  ("#fdeef1", "#c44a63", "#6b1f2f"),
}

LEGEND = [
    ("Python 包 / 实现", "py"),
    ("C++ 扩展源码", "cpp"),
    ("目录 / 分包", "pkg"),
    ("构建、CI、测试、文档", "bld"),
    ("外部依赖与补丁", "ext"),
]

# (path, note, kind, children)
TREE = (
    ("upstream/jax  ·  jax-ml/jax @ 2d66622 (jax-v0.11.1)",
     "JAX 主仓库：Python 包 + C++ 扩展 + PJRT 插件 + 构建/测试/文档", "root", (

        ("jax/", "Python 包：公共 API 与全部实现（714 个文件）", "pkg", (
            ("__init__.py", "公共 API 汇总；从 jax._src.* 转出并做弃用处理", "py", ()),
            ("_src/", "真正的实现层（388 个文件）", "pkg", (
                ("core.py", "内核：Tracer / Primitive / 抽象值 / Jaxpr / 效应（4428 行）", "py", ()),
                ("api.py", "用户变换：jit / grad / vmap / jvp / vjp / jacfwd / jacrev（2876 行）", "py", ()),
                ("pjit.py", "jit 的实现：分片解析、快速路径、C++ pjit 缓存（2655 行）", "py", ()),
                ("interpreters/", "变换与降级实现（7 个模块）", "pkg", (
                    ("ad.py", "自动微分：JVP / VJP / linearize / backward_pass", "py", ()),
                    ("batching.py", "vmap 变换：BatchTracer / 广播与规约规则", "py", ()),
                    ("partial_eval.py", "Jaxpr 追踪与部分求值（jit 的追踪前端）", "py", ()),
                    ("mlir.py", "jaxpr → MLIR：StableHLO + sdy 降级（3560 行）", "py", ()),
                    ("pxla.py", "mesh/分片执行、分片计算降级、结果处理（2106 行）", "py", ()),
                    ("remat.py", "重计算（grad checkpoint / offload）", "py", ()),
                )),
                ("lax/", "核心 primitives 及其 shape/dtype/分片/lowering 规则", "py", (
                    ("lax.py", "逐元素、reduction、dot、conv 等基础算子", "py", ()),
                    ("linalg.py", "线性代数（Cholesky/QR/SVD/eigh…），含 CPU/GPU lowering 注册", "py", ()),
                    ("parallel.py", "集合通信与 psum/pmax 等并行原语", "py", ()),
                    ("control_flow/", "cond / while / scan / for 循环原语", "py", ()),
                    ("pallas_lowerings/", "Pallas 的 lax 层降级规则", "py", ()),
                    ("special.py fft.py convolution.py slicing.py scaled_dot.py", "特殊函数、FFT、卷积、切片、scaled dot", "py", ()),
                )),
                ("numpy/", "NumPy 兼容前端（24 个模块：ufuncs、linalg、einsum、indexing…）", "py", ()),
                ("scipy/", "SciPy 兼容子集（47 个模块：sparse、optimize、signal、stats…）", "py", ()),
                ("nn/ ops/ random/ image/ cudnn/", "神经网络层、算子封装、随机数、图像、CuDNN 融合", "py", ()),
                ("pallas/", "自定义 kernel 框架（62 个文件）", "pkg", (
                    ("core.py pallas_call.py", "BlockSpec / grid / MemoryRef 与 pallas_call 入口", "py", ()),
                    ("mosaic/", "TPU 后端：Mosaic lowering、interpret、SC 原语", "py", ()),
                    ("mosaic_gpu/", "GPU 后端：Mosaic GPU lowering 与 pass pipeline", "py", ()),
                    ("triton/", "Triton 后端：Triton 原语与 kernel 生成", "py", ()),
                    ("fuser/", "自定义 fusion 基础库（fusible / jaxpr_fusion）", "py", ()),
                )),
                ("sharding_impls.py mesh.py layout.py op_shardings.py", "分片对象、设备网格、布局与算子分片规则", "py", ()),
                ("xla_bridge.py", "后端发现、PJRT 插件注册、平台规范化（1237 行）", "py", ()),
                ("compiler.py", "编译编排：选项、编译缓存、backend_compile_and_load", "py", ()),
                ("stages.py", "Traced / Lowering / Compiled 对象模型（jax.stages 的实体）", "py", ()),
                ("config.py", "全局配置项（约 111 个 jax.config 开关）", "py", ()),
                ("dtypes.py errors.py effects.py tree_util.py linear_util.py", "类型、错误、效应、PyTree、函数变换基础设施", "py", ()),
                ("array.py basearray.py earray.py indexing.py", "数组抽象、可扩展数组、索引（ds）", "py", ()),
                ("export/", "StableHLO 导出、序列化与 shape poly", "py", ()),
                ("state/ ref.py", "可变状态与内存引用（ref / discharge）", "py", ()),
                ("debugger/ profiler.py monitoring.py", "调试器、性能分析与事件记录", "py", ()),
                ("clusters/", "Slurm / K8s / MPI / Cloud TPU 集群接入", "py", ()),
                ("tpu/ cudnn/ third_party/", "TPU 专有线性代数、CuDNN、第三方 SciPy 补丁", "py", ()),
                ("lib/", "jaxlib 绑定与 MLIR dialects 装载", "pkg", ()),
                ("internal_test_util/", "测试用内部工具与导出向后兼容数据", "test", ()),
            )),
            ("numpy/ lax/ nn/ ops/ scipy/ random.py", "域 API 门面（薄转出到 _src，无顶层 fft.py/linalg.py）", "py", ()),
            ("core.py tree_util.py dtypes.py errors.py typing.py", "公共类型与兼容层", "py", ()),
            ("stages.py export.py sharding.py distributed.py ffi.py", "编译阶段、导出、分片、分布式、FFI 公共入口", "py", ()),
            ("extend/", "JAX 扩展 API（core/mlir/pallas/random/sharding/xla…）", "py", ()),
            ("experimental/", "实验特性（194 个文件）", "pkg", (
                ("pallas/ mosaic/", "Pallas 公共 API 与 Mosaic 方言封装", "py", ()),
                ("jax2tf/", "JAX ↔ TensorFlow 图互转", "py", ()),
                ("shard_map.py custom_partitioning.py pjit.py", "显式分片与自定义分区", "py", ()),
                ("sparse/ jet.py ode.py rnn.py checkify.py", "稀疏、Jet 高阶微分、ODE、RNN、checkify", "py", ()),
                ("array_serialization/ compilation_cache/", "数组序列化与持久编译缓存", "py", ()),
                ("colocated_python/ source_mapper/ roofline/", "Pathways 协同、源码映射、roofline 模型", "py", ()),
                ("multihost_utils.py transfer.py topologies.py", "多主机工具、传输、拓扑", "py", ()),
            )),
            ("interpreters/", "兼容 shim：转出到 jax._src.interpreters", "py", ()),
            ("tools/ example_libraries/ image/ lib/", "工具、示例库（stax/optimizers）、图像、字节码", "py", ()),
        )),

        ("jaxlib/", "C++ 扩展源码：绑定、运行时、后端 kernel（322 个文件）", "cpp", (
            ("jax_jit.cc  pjit.cc", "Python↔C++ 调用缓存、参数展平、快速分派", "cpp", ()),
            ("py_client.cc  py_executable.cc", "PJRT/IFRT 客户端与可执行对象（编译、执行、分片）", "cpp", ()),
            ("py_array.cc  py_device.cc  py_values.cc", "设备数组、设备、值对象及其缓冲区管理", "cpp", ()),
            ("xla_compiler.cc", "MLIR 模块编译入口；HLO 工具 hlo.cc 属 XLA 侧 xla/python", "cpp", ()),
            ("sharding.cc  partition_spec.cc  to_ifrt_sharding.cc", "分片与分区规格的 C++ 表示", "cpp", ()),
            ("pytree.cc  guard_lib.cc  config.cc  ffi.cc  callback.cc", "PyTree、防护、配置、外部函数接口、主机回调", "cpp", ()),
            ("cpu/", "CPU kernel：LAPACK、稀疏、三对角求解", "cpp", ()),
            ("gpu/ cuda/ rocm/ oneapi/", "GPU kernel、cuBLAS/cuSolver/cuSPARSE 绑定与插件扩展", "cpp", ()),
            ("mosaic/", "Mosaic GPU 方言与降级；TPU 方言本体在 XLA（此处为转发壳）", "cpp", ()),
            ("mlir/  triton/", "MLIR 与 Triton 的 Python 绑定/dialect", "cpp", ()),
            ("tools/", "各 wheel（jaxlib、cuda/rocm/oneapi、mosaic）打包脚本", "bld", ()),
        )),

        ("jax_plugins/", "PJRT 插件包：cuda / rocm / oneapi，通过 initialize() 注册", "cpp", ()),
        ("build/", "Bazel 构建入口（build.py）、requirements 锁、numpy.json", "bld", ()),
        ("ci/", "CI 脚本：CPU/CUDA/ROCm/TPU 的 Bazel 与 pytest 流水线", "bld", ()),
        ("tests/", "Python 测试（261 个文件，覆盖 API、变换、分片、导出）", "test", ()),
        ("benchmarks/", "性能基准（api、linalg、math、random、tracing、mosaic）", "test", ()),
        ("docs/", "文档与 notebook 教程，含 autodidax 与 internals", "test", ()),
        ("examples/", "示例：FFI、C++、k8s、ONNX、MNIST、SPMD", "test", ()),
        ("third_party/", "补丁与外部依赖封装（xla、absl、protobuf、grpc、rocm_wheels）", "ext", ()),
        ("MODULE.bazel  WORKSPACE", "Bazel 模块依赖：xla、stablehlo、shardy、triton、llvm", "bld", ()),
        ("setup.py  build_wheel.py  pyproject.toml", "wheel 打包与依赖声明（numpy/ml_dtypes/scipy/opt_einsum）", "bld", ()),
    )),
)


def render_tree() -> str:
  width = 1700
  row_h = 27
  top = 118
  indent = 32
  base_x = 34
  note_x = 660

  rows: list[tuple[int, str, str, str]] = []

  def walk(nodes, depth):
    for path, note, kind, children in nodes:
      rows.append((depth, path, note, kind))
      walk(children, depth + 1)

  walk(TREE, 0)

  body_h = len(rows) * row_h
  legend_h = 60
  height = top + body_h + legend_h + 30

  out = [
      f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
      f'height="{height}" viewBox="0 0 {width} {height}" '
      f'font-family="{", ".join(repr(f) for f in CJK)}">',
      '<rect width="100%" height="100%" fill="#ffffff"/>',
      f'<text x="{base_x}" y="46" font-size="27" font-weight="700" '
      f'fill="#12263f">JAX 仓库源码结构树 — jax-ml/jax @ jax-v0.11.1</text>',
      f'<text x="{base_x}" y="74" font-size="14" fill="#5a6a7d">'
      f'根路径 upstream/jax · commit 2d66622450e2c8633cda2307688ef7aa294bd6eb · '
      f'文件计数来自固定检出 · 括号内行数/文件数为实测</text>',
      f'<text x="{base_x}" y="96" font-size="13" fill="#8a94a6">'
      f'每个节点都是在固定检出中真实存在的路径；右侧说明为该路径的职责。</text>',
  ]

  # rows
  for i, (depth, path, note, kind) in enumerate(rows):
    y = top + i * row_h
    fill, accent, fg = KIND[kind]
    out.append(
        f'<rect x="{base_x}" y="{y + 2}" width="{width - 2 * base_x}" '
        f'height="{row_h - 4}" rx="4" fill="{fill}"/>')
    out.append(
        f'<rect x="{base_x}" y="{y + 2}" width="4" height="{row_h - 4}" '
        f'fill="{accent}"/>')
    tx = base_x + 12 + depth * indent
    size = 14 if depth <= 1 else 13
    weight = "700" if depth <= 1 else "500"
    out.append(
        f'<text x="{tx}" y="{y + row_h - 8}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fg}" '
        f'font-family="{", ".join(repr(f) for f in MONO)}">{esc(path)}</text>')
    if note:
      out.append(
          f'<text x="{note_x}" y="{y + row_h - 8}" font-size="13" '
          f'fill="#4a5568">{esc(note)}</text>')

  # connectors: one elbow bracket per parent, drawn in the indent gutter
  counter = 0
  child_rows: dict[int, list[int]] = {}

  def walk2(nodes, depth):
    nonlocal counter
    for _path, _note, _kind, children in nodes:
      me = counter
      counter += 1
      if children:
        first = counter
        walk2(children, depth + 1)
        child_rows[me] = list(range(first, counter))
    return

  walk2(TREE, 0)

  for parent_row, kids in child_rows.items():
    pdepth = rows[parent_row][0]
    parent_y = top + parent_row * row_h + row_h // 2
    last_y = top + kids[-1] * row_h + row_h // 2
    spine = base_x + 12 + (pdepth + 1) * indent - 14
    out.append(
        f'<path d="M {spine} {parent_y + row_h // 2 - 2} L {spine} {last_y}" '
        f'stroke="#b9c4d2" stroke-width="1.4" fill="none"/>')
    for kid in kids:
      ky = top + kid * row_h + row_h // 2
      kx = base_x + 12 + rows[kid][0] * indent - 4
      out.append(
          f'<path d="M {spine} {ky} L {kx} {ky}" stroke="#b9c4d2" '
          f'stroke-width="1.4" fill="none"/>')

  # legend
  ly = top + body_h + 26
  out.append(
      f'<text x="{base_x}" y="{ly}" font-size="13" font-weight="700" '
      f'fill="#39424f">图例</text>')
  lx = base_x + 60
  for label, kind in LEGEND:
    fill, accent, _ = KIND[kind]
    out.append(f'<rect x="{lx}" y="{ly - 12}" width="16" height="14" rx="3" '
               f'fill="{fill}" stroke="{accent}"/>')
    out.append(f'<text x="{lx + 22}" y="{ly}" font-size="13" '
               f'fill="#4a5568">{esc(label)}</text>')
    lx += 230

  out.append('</svg>')
  return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Diagram 2: upstream / downstream chain
# ---------------------------------------------------------------------------

BAND = ("#f7f9fc", "#dbe4ee")

CHAIN = [
    ("上游依赖 · UPSTREAM", "#5b3a8e", [
        ("xla (openxla/xla)", "编译与运行时\nxla/python · service · pjrt"),
        ("stablehlo", "可移植算子集\nJAX 的 MLIR 出口方言"),
        ("shardy", "张量分片传播\nsdy 方言与 partitioner"),
        ("llvm-project", "CPU/GPU 代码生成\nLLVM IR → 机器码"),
        ("triton", "GPU kernel DSL\nPallas-Triton 后端"),
        ("Mosaic", "TPU 方言在 XLA\nGPU 方言在 jaxlib"),
        ("Python 依赖", "numpy · ml_dtypes\nscipy · opt_einsum"),
    ]),
    ("JAX Python 公共 API · USER API", "#2f6fb0", [
        ("jax.jit / jax.grad / jax.vmap", "jax/_src/api.py\n用户变换入口"),
        ("jax.numpy · jax.lax · jax.nn", "jax/_src/numpy · lax · nn\n域前端与算子"),
        ("jax.sharding · jax.export", "jax/_src/sharding_impls.py\nexport/_export.py"),
        ("jax.experimental.pallas", "jax/_src/pallas/pallas_call.py\n自定义 kernel"),
        ("jax.extend · jax.stages", "无兼容保证的扩展与阶段对象"),
    ]),
    ("追踪与变换 · TRACING & TRANSFORMS", "#2f9e6b", [
        ("core.Tracer / Primitive.bind", "jax/_src/core.py\n抽象值 + 追踪器内核"),
        ("partial_eval", "jax/_src/interpreters/partial_eval.py\n追踪为 Jaxpr（jit 前端）"),
        ("ad", "jax/_src/interpreters/ad.py\nJVP / VJP / linearize"),
        ("batching", "jax/_src/interpreters/batching.py\nvmap 变换"),
        ("pjit + pxla", "jax/_src/pjit.py · interpreters/pxla.py\n分片解析与网格执行"),
    ]),
    ("IR 与降级 · LOWERING", "#c07a2b", [
        ("core.Jaxpr", "jax/_src/core.py\n平台无关中间表示"),
        ("mlir.lower_jaxpr_to_module", "jax/_src/interpreters/mlir.py:1327\njaxpr → MLIR 模块"),
        ("StableHLO + sdy", "稳定算子集 + 分片标注\nuse_shardy_partitioner 开关"),
        ("MLIR pass pipeline", "sdy-lift-inlined-meshes\n内建与 JAX 注册的 pass"),
        ("Stages: Lowered / Compiled", "jax/_src/stages.py\nlower() / compile() 对象"),
    ]),
    ("jaxlib C++ 扩展 · C++ BINDINGS", "#8a4b08", [
        ("jax_jit.cc / pjit.cc", "调用缓存、参数展平\n快速分派路径"), 
        ("xla_compiler.cc", "MLIR 模块 → XLA 编译"),
        ("py_client.cc", "IFRT/PJRT 客户端\nCompile / CompileAndLoad"),
        ("py_executable.cc / py_array.cc", "可执行对象与设备数组"),
        ("cpu/ gpu/ cuda/ rocm/ oneapi", "后端 kernel 与厂商库绑定"),
    ]),
    ("XLA 编译与运行时 · XLA", "#c44a63", [
        ("xla/python", "Python 绑定：xla_builder\nhlo · ops · profiler · ifrt"),
        ("xla/service HLO passes", "优化、布局、fusion\nShardy partitioner"),
        ("xla/pjrt runtime", "PjRtClient / PjRtExecutable\n设备内存与执行"),
        ("xla/backends · xla/tpu", "cpu · gpu · interpreter\nTPU 相关在 xla/tpu"),
        ("PJRT 插件", "jax_plugins 的 initialize()\nlibtpu / xla_cuda_plugin"),
    ]),
    ("硬件 · HARDWARE", "#39424f", [
        ("CPU", "x86 / ARM\noneDNN · XNNPACK · Eigen"),
        ("NVIDIA GPU", "CUDA · cuBLAS/cuDNN\ncuSPARSE · cuSolver"),
        ("AMD GPU", "ROCm · hipBLAS"),
        ("Intel GPU", "oneAPI · SyCL"),
        ("TPU", "libtpu · Mosaic\nPallas TPU kernel"),
    ]),
    ("下游消费者 · DOWNSTREAM", "#12395e", [
        ("建模框架", "Flax · Equinox · Haiku\nKeras 3 · JAX-CNN"),
        ("训练与优化", "Optax · Orbax\nLevanter · MaxText · T5X"),
        ("推理与服务", "SGLang-JAX · vLLM-JAX\nPathways · jax2tf/TF"),
        ("生态工具", "jaxtyping · jaxlie\nblackjax · numpyro"),
    ]),
]


def render_pipeline() -> str:
  width = 1760
  left = 40
  right = width - 40
  band_gap = 16
  header_h = 30
  box_h = 92
  band_h = header_h + box_h + 16
  n_bands = len(CHAIN)
  height = 104 + (n_bands - 1) * (band_h + band_gap) + band_h + 28

  out = [
      f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
      f'height="{height}" viewBox="0 0 {width} {height}" '
      f'font-family="{", ".join(repr(f) for f in CJK)}">',
      '<rect width="100%" height="100%" fill="#ffffff"/>',
      f'<text x="{left}" y="46" font-size="27" font-weight="700" '
      f'fill="#12263f">JAX 上下游链路 — 从用户函数到硬件执行</text>',
      f'<text x="{left}" y="74" font-size="14" fill="#5a6a7d">'
      f'上行为上游依赖，中部为 JAX 自身的处理链，下行为下游消费者；'
      f'标注为关键源码位置（相对 upstream/jax 或 upstream/xla）。</text>',
  ]

  y = 104
  for title, accent, boxes in CHAIN:
    n = len(boxes)
    gap = 14
    bw = (right - left - header_h - gap * (n - 1)) / n
    out.append(
        f'<rect x="{left}" y="{y}" width="{right - left}" '
        f'height="{band_h}" rx="10" fill="{BAND[0]}" '
        f'stroke="{BAND[1]}"/>')
    out.append(f'<rect x="{left}" y="{y}" width="6" '
               f'height="{band_h}" rx="3" fill="{accent}"/>')
    out.append(
        f'<text x="{left + 16}" y="{y + 21}" font-size="14" font-weight="700" '
        f'fill="{accent}">{esc(title)}</text>')

    bx = left + header_h + 4
    for btitle, bsub in boxes:
      out.append(
          f'<rect x="{bx:.1f}" y="{y + header_h - 6}" width="{bw:.1f}" '
          f'height="{box_h}" rx="8" fill="#ffffff" stroke="{accent}" '
          f'stroke-opacity="0.45"/>')
      out.append(
          f'<text x="{bx + 12:.1f}" y="{y + header_h + 24}" font-size="13.5" '
          f'font-weight="700" fill="#1f2b3a">{esc(btitle)}</text>')
      for j, line in enumerate(bsub.split("\n")):
        out.append(
            f'<text x="{bx + 12:.1f}" y="{y + header_h + 48 + j * 17}" '
            f'font-size="11.5" fill="#63718a" '
            f'font-family="{", ".join(repr(f) for f in MONO)}">'
            f'{esc(line)}</text>')
      bx += bw + gap

    # downward arrow into the next band
    mid = (left + right) / 2
    ay = y + band_h
    out.append(
        f'<path d="M {mid} {ay} L {mid} {ay + band_gap}" stroke="{accent}" '
        f'stroke-width="2.4" marker-end="url(#arrow)"/>')
    y = ay + band_gap

  out.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M 0 0 L 10 5 L 0 10 z" fill="#5a6a7d"/></marker></defs>')
  out.append('</svg>')
  return "\n".join(out) + "\n"


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "jax-repo-structure.svg").write_text(render_tree(), encoding="utf-8")
  (OUT / "jax-repo-pipeline.svg").write_text(render_pipeline(), encoding="utf-8")
  print("wrote", OUT / "jax-repo-structure.svg")
  print("wrote", OUT / "jax-repo-pipeline.svg")


if __name__ == "__main__":
  main()
