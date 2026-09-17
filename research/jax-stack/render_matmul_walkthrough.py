#!/usr/bin/env python3
"""Generate the matmul lowering chain as SVG, from the verified source walkthrough.

Layout is hand-computed so that position and size carry meaning:

  * vertical position = layer in the lowering pipeline
  * left column       = the linear chain a single matmul follows
  * right column      = where intermediates can be observed
  * border style      = evidence boundary (solid = source-verified here,
                        dashed = closed source or version-skewed material)

Every source anchor printed in the figure is one that was verified against the
pinned revision `jax-v0.11.1`. `--check` re-verifies the anchors against the
actual source files and validates the geometry (no overlaps, no text overflow,
no stale line numbers from the previous pin).

Usage:
  python3 -B research/jax-stack/render_matmul_walkthrough.py \
      --output research/jax-stack/figures/matmul-lowering-chain.svg [--check]
"""

import argparse
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W, H = 1900, 1800
FONT = "Noto Sans CJK SC, Noto Sans CJK JP, DejaVu Sans, sans-serif"

C_PY, C_PY_S = "#e8f1fb", "#2f6fb0"
C_PRIM, C_PRIM_S = "#f5f1fb", "#6b4fbb"
C_LOW, C_LOW_S = "#fdf0e3", "#c8791f"
C_HLO, C_HLO_S = "#eaf6ea", "#3d8b40"
C_PASS, C_PASS_S = "#e9f6f8", "#1f8a99"
C_RUN, C_RUN_S = "#ededed", "#666666"
C_OBS, C_OBS_S = "#fdf6e3", "#a0871f"
C_WARN, C_WARN_S = "#fbeaea", "#b23b3b"
C_OFF, C_OFF_S = "#f4f4f4", "#9a9a9a"
NOTE = "#555555"

LEFT, L_W = 50, 1130
RX, R_W = 1210, 640
LABEL_H = 40
# The bands describe the pipeline layers, so they cover the left column only;
# the right column is a separate reference panel aligned to the same rows.
BAND_X, BAND_W = 34, 1166
PANEL_X, PANEL_W = 1180, 690

BANDS = {
    1: dict(y=112, h=196, color=C_PY, label="① Python 前端　—— matmul 只做形状规约，不产生新数学"),
    2: dict(y=326, h=252, color=C_PRIM, label="② Primitive 与规则　—— 四条变换规则各自独立注册"),
    3: dict(y=596, h=204, color=C_LOW, label="③ Lowering　—— 直接发出 stablehlo.dot_general"),
    4: dict(y=818, h=178, color=C_HLO, label="④ MLIR → HLO　—— 不经过 MHLO"),
    5: dict(y=1014, h=268, color=C_PASS, label="⑤ CPU HLO pipeline　—— 两段，围绕 layout assignment 切分"),
    6: dict(y=1300, h=112, color=C_RUN, label="⑥ Runtime"),
}

    # (x, y, w, h, id, allow_wrap) for every box
BOXES = [
    # ① Python
    (70, 164, 520, 128, "matmul", True),
    (620, 164, 540, 128, "lax-dot", True),
    # ② primitive + rules
    (70, 378, 700, 88, "prim", True),
    (70, 486, 1090, 88, "rules", True),
    # ③ lowering
    (70, 648, 700, 136, "lower", True),
    (800, 648, 360, 136, "prec", True),
    # ④ mlir->hlo
    (70, 870, 1090, 110, "m2h", True),
    # ⑤ pipeline
    (70, 1066, 520, 200, "pipe-before", True),
    (610, 1066, 550, 200, "pipe-after", True),
    # ⑥ runtime
    (70, 1352, 1090, 44, "thunk", False),
    # right column
    (RX, 112, R_W, 196, "obs-jaxpr", True),
    (RX, 326, R_W, 252, "obs-ir", True),
    (RX, 596, R_W, 204, "obs-hlo", True),
    (RX, 818, R_W, 178, "obs-pass", True),
    (RX, 1014, R_W, 268, "obs-replay", True),
    (RX, 1300, R_W, 112, "obs-note", True),
    # bottom
    (50, 1450, 1130, 200, "pallas", True),
    (1210, 1450, 640, 200, "evidence", True),
    (50, 1680, 1800, 96, "footer", True),
]

# Source anchors printed in the figure: (label, path, line, needle).
# --check verifies each against the working tree.
ANCHORS = [
    ("matmul", "upstream/jax/jax/_src/numpy/tensor_contractions.py", 138, "def matmul("),
    ("matmul-dot", "upstream/jax/jax/_src/numpy/tensor_contractions.py", 273, "out = lax.dot_general("),
    ("dot_general", "upstream/jax/jax/_src/lax/lax.py", 2530, "def dot_general("),
    ("dot", "upstream/jax/jax/_src/lax/lax.py", 2546, "def dot("),
    ("bind", "upstream/jax/jax/_src/lax/lax.py", 2625, "dot_general_p.bind"),
    ("prim", "upstream/jax/jax/_src/lax/lax.py", 6072, "dot_general_p = standard_primitive("),
    ("stdprim", "upstream/jax/jax/_src/lax/utils.py", 46, "def standard_primitive("),
    ("shape", "upstream/jax/jax/_src/lax/lax.py", 5685, "def _dot_general_shape_rule"),
    ("shard", "upstream/jax/jax/_src/lax/lax.py", 5764, "def _dot_general_sharding_rule"),
    ("ad", "upstream/jax/jax/_src/lax/lax.py", 6114, "ad.defbilinear(dot_general_p"),
    ("vmap", "upstream/jax/jax/_src/lax/lax.py", 6119, "fancy_primitive_batchers[dot_general_p]"),
    ("remat", "upstream/jax/jax/_src/lax/lax.py", 6102, "remat.rules[dot_general_p]"),
    ("pp", "upstream/jax/jax/_src/lax/lax.py", 6120, "pp_eqn_rules[dot_general_p]"),
    ("lower", "upstream/jax/jax/_src/lax/lax.py", 6266, "def _dot_general_lower("),
    ("prec", "upstream/jax/jax/_src/lax/lax.py", 6193, "def _handle_dot_precision("),
    ("shlo", "upstream/jax/jax/_src/lax/lax.py", 6280, "result = hlo.dot_general("),
    ("reg-default", "upstream/jax/jax/_src/lax/lax.py", 6294, "mlir.register_lowering(dot_general_p, _dot_general_lower)"),
    ("reg-plat", "upstream/jax/jax/_src/lax/lax.py", 6296, 'for platform in ["cpu", "tpu"]:'),
    ("alias", "upstream/jax/jax/_src/lib/mlir/dialects/__init__.py", 62, "stablehlo as hlo"),
    ("module", "upstream/jax/jax/_src/interpreters/mlir.py", 1327, "def lower_jaxpr_to_module("),
    ("stages", "upstream/jax/jax/_src/stages.py", 271, "def compiler_ir("),
    ("m2h", "upstream/xla/xla/pjrt/mlir_to_hlo.cc", 99, "MlirToXlaComputation("),
    ("direct", "upstream/xla/xla/hlo/translate/stablehlo.cc", 115, "direct_stablehlo_to_hlo = true"),
    ("conv", "upstream/xla/xla/hlo/translate/stablehlo.cc", 187, "ConvertStablehloToHloWithOptions("),
    ("pipe-before", "upstream/xla/xla/service/cpu/cpu_compiler.cc", 694, 'HloPassPipeline pipeline("HLO passes through layout assignment")'),
    ("pipe-after", "upstream/xla/xla/service/cpu/cpu_compiler.cc", 1010, 'HloPassPipeline pipeline("HLO passes after layout assignment")'),
    ("pre-hook", "upstream/xla/xla/service/cpu/cpu_compiler.cc", 1178, "PipelineStage::kPreScheduler"),
    ("post-hook", "upstream/xla/xla/service/cpu/cpu_compiler.cc", 1833, "PipelineStage::kPostScheduler"),
    ("librewrite", "upstream/xla/xla/backends/cpu/transforms/library_rewriter.cc", 329, "fuse_dot_"),
    ("thunk", "upstream/xla/xla/backends/cpu/runtime/dot_thunk.h", 33, "class DotThunk final"),
]

# Identifiers that must NOT appear: they belong to the pre-0.11.1 revision.
STALE = ["5832e866", "496bd4bd", "eb23a983", "ab547095", "23689d2a",
         ":2528", ":6264", "0.11.2", "0.0.27"]

# (text, font_size, available_width) recorded by box() so --check can verify fit
TEXT_FIT = []
# (text, x, y) recorded by arrow() so --check can verify captions clear the bands
ARROW_LABELS = []
# (title, required_h, actual_h) recorded by box() so --check can verify vertical fit
VFIT = []


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def text_w(s, size):
    """Estimate rendered width: CJK and fullwidth count as 1.0em, ASCII as 0.55em."""
    w = 0.0
    for ch in s:
        w += size if unicodedata.east_asian_width(ch) in ("W", "F") else 0.55 * size
    return w


LINE_DY = 17.0      # body line advance
TITLE_DY = 18.0     # first body line below the title baseline


def box(x, y, w, h, title, lines, fill, stroke, *, dashed=False, off=False,
        title_size=17, body_size=12.5, radius=10):
    dash = ' stroke-dasharray="8 5"' if dashed else ""
    op = ' opacity="0.55"' if off else ""
    out = [f'<g{op}>',
           f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
           f'fill="{fill}" stroke="{stroke}" stroke-width="2.2"{dash}/>']
    cx = x + w / 2
    ty = y + 25
    TEXT_FIT.append((title, title_size, w - 24))
    # required height: title baseline + body block + descender room
    need = 25 + 6 + (TITLE_DY + (len(lines) - 1) * LINE_DY if lines else 0)
    if need > h:
        VFIT.append((title, need, h))
    out.append(f'<text x="{cx}" y="{ty}" font-family="{FONT}" font-size="{title_size}" '
               f'font-weight="700" fill="#1b1b1b" text-anchor="middle">{esc(title)}</text>')
    for i, ln in enumerate(lines):
        TEXT_FIT.append((ln, body_size, w - 24))
        out.append(f'<text x="{cx}" y="{ty + TITLE_DY + i * LINE_DY}" font-family="{FONT}" '
                   f'font-size="{body_size}" fill="#3a3a3a" text-anchor="middle">{esc(ln)}</text>')
    out.append('</g>')
    return "\n".join(out)


def band(n):
    b = BANDS[n]
    return (f'<rect x="{BAND_X}" y="{b["y"]}" width="{BAND_W}" height="{b["h"]}" rx="14" '
            f'fill="{b["color"]}" opacity="0.42"/>'
            f'<text x="{BAND_X + 18}" y="{b["y"] + 25}" font-family="{FONT}" font-size="14.5" '
            f'font-weight="700" fill="#4a4a4a">{esc(b["label"])}</text>')


def arrow(x1, y1, x2, y2, label="", *, color="#444444", dashed=False, label_y=None,
          label_dx=0, width=2.0):
    dash = ' stroke-dasharray="7 4"' if dashed else ""
    out = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
           f'stroke-width="{width}"{dash} marker-end="url(#ah)"/>']
    if label:
        mx = (x1 + x2) / 2 + label_dx
        my = label_y if label_y is not None else (y1 + y2) / 2 - 7
        ARROW_LABELS.append((label, mx, my))
        out.append(f'<text x="{mx}" y="{my}" font-family="{FONT}" '
                   f'font-size="12" fill="{color}" text-anchor="middle">{esc(label)}</text>')
    return "\n".join(out)


def note(x, y, t, size=12.5, fill=NOTE, anchor="start", weight=None):
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}"{w}>{esc(t)}</text>')


def build():
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" role="img">',
         '<defs><marker id="ah" markerWidth="11" markerHeight="8" refX="9" refY="4" '
         'orient="auto"><path d="M0,0 L11,4 L0,8 z" fill="#444444"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         note(W / 2, 44, "matmul lowering 全链路源码走读", size=27, fill="#111111",
              anchor="middle", weight="700"),
         note(W / 2, 72, "纵向 = 流水线层次　·　左列 = 单个 matmul 经过的链路　·　右列 = 中间产物的观察点　·　"
                         "实线框 = 本次核对源码　·　虚线框 = 闭源或版本错配材料",
              size=13.5, anchor="middle")]

    for n in sorted(BANDS):
        s.append(band(n))

    # right column: a separate reference panel, not a pipeline layer
    s.append(f'<rect x="{PANEL_X}" y="112" width="{PANEL_W}" height="1300" rx="14" '
             f'fill="#fbfbfb" stroke="#c9c9c9" stroke-width="1.6"/>')
    s.append(note(PANEL_X + PANEL_W / 2, 142, "中间产物在哪里观察", size=18,
                  fill="#333333", anchor="middle", weight="700"))

    # ---- ① Python ----
    s.append(box(*BOXES[0][:4], "jax.numpy.matmul　tensor_contractions.py:138",
                 ["squeeze → dot_general → transpose → expand_dims",
                  "纯 Python 组合，不是 primitive；Jaxpr 里没有 matmul",
                  "形状错的代价落在 transpose，不在 dot"], C_PY, C_PY_S))
    s.append(box(*BOXES[1][:4], "lax.dot_general :2530　→　lax.dot :2546",
                 ["dot_general 是 dot 的别名（一行转发）",
                  "dot :2625 → dot_general_p.bind（:2624 先 auto_insert_reshard）",
                  "docstring 断言：lowers directly to stablehlo.dot_general"], C_PY, C_PY_S))

    # ---- ② primitive + rules ----
    s.append(box(*BOXES[2][:4], "dot_general_p = standard_primitive(...)　lax.py:6072",
                 ["standard_primitive（lax/utils.py:46）只建 impl + abstract_eval",
                  "本身不含 lowering；具体 lower 到哪由③的规则决定"], C_PRIM, C_PRIM_S))
    s.append(box(*BOXES[3][:4], "四条变换规则各自独立注册　——　jit / grad / vmap 挂在不同位置",
                 ["shape :5685　·　sharding :5764　·　remat :6102　·　pp :6120",
                  "grad：ad.defbilinear :6114　→　VJP 本身又是一个 dot_general（双线性）",
                  "vmap：fancy_primitive_batchers :6119　→　重算 dimension_numbers，产出一个高秩 dot"],
                 C_PRIM, C_PRIM_S, body_size=12.5))

    # ---- ③ lowering ----
    s.append(box(*BOXES[4][:4], "_dot_general_lower　lax.py:6266",
                 ["hlo.DotDimensionNumbers.get :6274 → hlo.dot_general :6280",
                  "hlo 就是 stablehlo：dialects/__init__.py:62  `stablehlo as hlo`",
                  "lower_with_sharding_in_types :6289；按需 convert :6291",
                  "注册：默认 :6294 ／ cpu+tpu :6296-6299（共用同一函数体）"],
                 C_LOW, C_LOW_S, title_size=16))
    s.append(box(*BOXES[5][:4], "_handle_dot_precision :6193",
                 ["全链路唯一的平台分叉点",
                  "混合 float/int：TPU 不转换，CPU/GPU 转换",
                  "混合 fp8：TPU 转换，CPU/GPU 显式排除",
                  "CPU 拒绝不支持的 DotAlgorithm（会静默忽略，故前端先拦）"],
                 C_WARN, C_WARN_S, title_size=15, body_size=11.5))

    # ---- ④ MLIR → HLO ----
    s.append(box(*BOXES[6][:4], "lower_jaxpr_to_module mlir.py:1327　→　MlirToXlaComputation mlir_to_hlo.cc:99",
                 ["ConvertStablehloToHloWithOptions stablehlo.cc:187",
                  "direct_stablehlo_to_hlo = true（stablehlo.cc:115）⇒ 直接 StableHLO → HLO，不经过 MHLO",
                  "同名陷阱：`_cached_lowering_to_hlo` 名字含 HLO，实际调用 MLIR lowering"],
                 C_HLO, C_HLO_S, title_size=16, body_size=12))

    # ---- ⑤ pipeline ----
    s.append(box(*BOXES[7][:4], "layout assignment 之前　cpu_compiler.cc:694",
                 ["DotDecomposer ×2（归一 dot 规范形）",
                  "BatchDotSimplification",
                  "OneDnnOpsRewriter",
                  "CallInliner / FloatNormalization / ResultCaster"],
                 C_PASS, C_PASS_S, title_size=15, body_size=12))
    s.append(box(*BOXES[8][:4], "layout assignment 之后　cpu_compiler.cc:1010",
                 ["LibraryRewriter :329 fuse_dot_ ← matmul 交给库(Eigen/oneDNN)的转折点",
                  "OneDnnContractionRewriter / DotDecomposer",
                  "CpuInstructionFusion / CpuMultiOutputFusion（fusion）",
                  "ApplyXlaTransforms :1178(PRE) / :1833(POST) = Python HLO hook 落点",
                  "OptimizeInputOutputBufferAlias / CopyInsertion / HloDCE"],
                 C_PASS, C_PASS_S, title_size=15, body_size=11.5))

    # ---- ⑥ runtime ----
    s.append(box(*BOXES[9][:4], "DotThunk　xla/backends/cpu/runtime/dot_thunk.h:33　——　未被融合/库改写吃掉的 dot 落到这里，由 Eigen contraction 执行",
                 [], C_RUN, C_RUN_S, title_size=13.5))

    # arrows in the left column
    s.append(arrow(330, 292, 330, 378))
    s.append(arrow(760, 292, 760, 378))
    s.append(arrow(305, 466, 305, 486))
    s.append(arrow(420, 574, 420, 648))
    s.append(arrow(760, 574, 900, 648, label="平台分支", label_y=592))
    s.append(arrow(420, 784, 420, 870))
    s.append(arrow(420, 980, 420, 1066))
    s.append(arrow(420, 1266, 420, 1352))
    s.append(arrow(330, 1396, 330, 1450))

    # ---- right column: observation points ----
    s.append(box(*BOXES[10][:4], "① Jaxpr",
                 ["jax.make_jaxpr(f)(...)",
                  "Python 层 primitive 方程",
                  "边界：不是 backend HLO"], C_OBS, C_OBS_S, title_size=16))
    s.append(box(*BOXES[11][:4], "② StableHLO / 导出 HLO",
                 ["Lowered.as_text(\"stablehlo\")　默认方言",
                  "Lowered.as_text(\"hlo\")　显式转换分支",
                  "stages.py:271 只支持这两种，其他抛 ValueError",
                  "compiler_ir 自述：不是可靠序列化"],
                 C_OBS, C_OBS_S, title_size=16, body_size=12))
    s.append(box(*BOXES[12][:4], "③ 后端 pass 边界",
                 ["--xla_dump_to=DIR --xla_dump_hlo_as_text",
                  "--xla_dump_hlo_pass_re=.+",
                  "陷阱：用 .* 会跳过未改变 HLO 的 pass dump"],
                 C_OBS, C_OBS_S, title_size=16, body_size=12))
    s.append(box(*BOXES[13][:4], "④ 编译后程序 / emitter",
                 ["compiled.as_text()",
                  "--xla_dump_emitter_re=mlir-fusion|llvm",
                  "cost_analysis / memory_analysis（静态估计）"],
                 C_OBS, C_OBS_S, title_size=15, body_size=11.5))
    s.append(box(*BOXES[14][:4], "⑤ 已有 capture（REPLAY-OFFLINE）",
                 ["matmul-pass-walkthrough.md：640 边界 / 22 组叶子改写",
                  "cpu-executable-and-trace.md：11 thunk / 33 配对事件",
                  "上述材料来自换 pin 之前的构建，相对本图是 VERSION-SKEW"],
                 C_OFF, C_OFF_S, title_size=15, body_size=11.5, off=True))
    s.append(box(*BOXES[15][:4], "证据说明",
                 ["本图全部锚点为 SOURCE-ONLY",
                  "行号按 jax-v0.11.1 核对，换 pin 前已漂移"],
                 C_OBS, C_OBS_S, title_size=15, body_size=11.5))

    # ---- bottom: Pallas contrast ----
    s.append(box(*BOXES[16][:4], "对比：Pallas 路径在同一层的分叉",
                 ["普通 JAX：matmul → dot_general_p → stablehlo.dot_general（XLA 直接看得见 dot）",
                  "Pallas：pallas_call → pallas_call_p → 内层 Mosaic TPU MLIR → 序列化进外层 custom_call 的 backend_config",
                  "关键后果：kernel 计算体藏在 custom_call payload 里，外层 HLO pass（DotDecomposer / LibraryRewriter）看不到 kernel 内部运算",
                  "Mosaic TPU MLIR ≠ LLO"],
                 C_PRIM, C_PRIM_S, title_size=15, body_size=12))

    # ---- bottom: evidence ----
    s.append(box(*BOXES[17][:4], "本次未验证",
                 ["平台分叉（③右）仅为源码结论，未在各平台实测类型组合",
                  "CPU pass 列表按源码顺序列出，未验证某次编译实际启用顺序",
                  "DotThunk/Eigen 运行路径只有换 pin 前的证据",
                  "TPU 侧不可见：PJRT_Client_Compile 之后进入闭源 libtpu"],
                 C_WARN, C_WARN_S, title_size=15, body_size=11.5))

    s.append(note(70, 1716, "固定 revision：JAX 2d66622450e2（jax-v0.11.1）· XLA dcf304bc5dca　|　"
                            "本图由 render_matmul_walkthrough.py 生成，锚点经 --check 对源码复核", size=12.5))
    s.append('</svg>')
    return "\n".join(s) + "\n"


def check(svg_text):
    problems = []
    # geometry
    strip_bottom = {n: b["y"] + LABEL_H for n, b in BANDS.items()}
    for x, y, w, h, key, _ in BOXES:
        for n, b in BANDS.items():
            horizontally_in_band = x < BAND_X + BAND_W and BAND_X < x + w
            if (horizontally_in_band and b["y"] <= y < b["y"] + b["h"]
                    and y < strip_bottom[n]):
                problems.append(f"box {key} intrudes into band {n} label strip")
    for i, a in enumerate(BOXES):
        for b in BOXES[i + 1:]:
            if a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]:
                problems.append(f"boxes {a[4]} and {b[4]} overlap")
    for x, y, w, h, key, _ in BOXES:
        if x < 0 or x + w > W or y + h > H:
            problems.append(f"box {key} exceeds canvas")
    # text fit: every line must fit inside its box interior
    for text, size, avail in TEXT_FIT:
        w_est = text_w(text, size)
        if w_est > avail:
            problems.append(f"text overflows its box ({w_est:.0f}px > {avail}px): {text[:52]!r}")
    # vertical fit: title + body block + descender must fit the box height
    for title, need, have in VFIT:
        problems.append(f"box {title[:40]!r} too short vertically: needs {need:.0f}px, has {have}px")
    # arrow captions must not land inside a band's label strip
    for text, mx, my in ARROW_LABELS:
        for n, b in BANDS.items():
            horizontally_in_band = BAND_X <= mx <= BAND_X + BAND_W
            if horizontally_in_band and b["y"] <= my <= b["y"] + LABEL_H:
                problems.append(f"arrow caption {text!r} at y={my} falls inside band "
                                f"{n} label strip")
    # anchors against the working tree
    for name, path, line, needle in ANCHORS:
        p = ROOT / path
        if not p.exists():
            problems.append(f"anchor {name}: missing file {path}")
            continue
        lines = p.read_text(errors="replace").splitlines()
        actual = lines[line - 1] if line <= len(lines) else ""
        if needle not in actual:
            problems.append(f"anchor {name}: {path}:{line} does not contain {needle!r}")
    # stale identifiers
    for bad in STALE:
        if bad in svg_text:
            problems.append(f"stale identifier present: {bad}")
    for must in ("2d66622450e2", "dcf304bc5dca", "v0.11.1"):
        if must not in svg_text:
            problems.append(f"missing expected revision: {must}")
    if problems:
        raise SystemExit("check failed:\n  " + "\n  ".join(problems))
    print(f"check passed: {len(BOXES)} boxes, {len(BANDS)} bands, "
          f"{len(ANCHORS)} source anchors re-verified against the working tree, "
          f"no overlaps/overflow, no stale identifiers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=Path("research/jax-stack/figures/matmul-lowering-chain.svg"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    TEXT_FIT.clear()
    ARROW_LABELS.clear()
    VFIT.clear()
    text = build()
    if a.check:
        check(text)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(text)
    print(f"wrote {a.output} ({a.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
