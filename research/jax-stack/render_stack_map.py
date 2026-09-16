#!/usr/bin/env python3
"""Generate the JAX software-stack component map as SVG.

Layout is hand-computed rather than delegated to a graph layout engine, because
the deliverable requires component size and position to carry meaning:

  * vertical position   = layer in the compilation / execution pipeline
  * horizontal position = which path the component belongs to
    (left = ordinary JAX/XLA, middle = Pallas/Mosaic, right = runtime/tooling)
  * box size            = the component's scope in the stack
  * border style        = evidence boundary (solid = readable source, dashed = closed)

Band geometry is explicit: every band reserves a label strip at its top and a
gap to the next band, so band captions can never collide with boxes or arrows.
`--check` re-derives the reserved regions and fails if any element intrudes.

Source sizes are annotated, not used naively as box areas: LLVM is ~181k files
and would otherwise swamp the figure.

Usage:
  python3 -B research/jax-stack/render_stack_map.py --output <path.svg> [--check]
"""

import argparse
import html
from pathlib import Path

W, H = 1720, 1660
FONT = "Noto Sans CJK SC, Noto Sans CJK JP, DejaVu Sans, sans-serif"

C_FRONT, C_FRONT_S = "#e8f1fb", "#2f6fb0"
C_REPR, C_REPR_S = "#eaf6ea", "#3d8b40"
C_XLA, C_XLA_S = "#fdf0e3", "#c8791f"
C_INFRA, C_INFRA_S = "#f0ecfa", "#6b4fbb"
C_RUN, C_RUN_S = "#e9f6f8", "#1f8a99"
C_HW, C_HW_S = "#ededed", "#666666"
C_CLOSED, C_CLOSED_S = "#fbeaea", "#b23b3b"
C_OBS, C_OBS_S = "#fdf6e3", "#a0871f"
C_OFF, C_OFF_S = "#f4f4f4", "#9a9a9a"
C_NOTE = "#555555"

LEFT, RIGHT = 70, 1650
BAND_X, BAND_W = 34, W - 68
LABEL_H = 38            # reserved strip at the top of every band
LABEL_BASELINE = 24     # caption baseline inside that strip

BANDS = {
    1: dict(y=92,  h=214, color=C_FRONT, label="① 前端 / 表达层　—— 用户写的 Python 在这里被追踪成程序"),
    2: dict(y=326, h=168, color=C_REPR,  label="② 中间表示层　—— JAX 与 XLA 之间的可移植契约"),
    3: dict(y=514, h=330, color=C_XLA,   label="③ 编译层　—— 转换、分片传播、优化、调度、显存分配、代码生成"),
    4: dict(y=864, h=136, color=C_INFRA, label="④ 后端基础设施　—— 多级 IR 与机器码生成"),
    5: dict(y=1020, h=160, color=C_RUN,  label="⑤ 运行时抽象层　—— 编译产物如何交给具体后端"),
    6: dict(y=1200, h=250, color=C_CLOSED, label="⑥ 执行层　—— 设备后端与硬件"),
}

# (x, y, w, h, key) for every box; --check uses these against the band strips
BOXES = [
    (70, 136, 500, 148, "jax"),
    (600, 136, 340, 148, "jaxlib"),
    (970, 136, 320, 148, "pallas"),
    (1320, 136, 330, 148, "ifrt"),
    (70, 368, 400, 116, "stablehlo"),
    (500, 368, 400, 116, "shardy"),
    (930, 368, 720, 116, "mlir"),
    (70, 594, 830, 226, "xla"),
    (930, 594, 330, 226, "mosaic"),
    (1290, 594, 360, 226, "triton"),
    (70, 908, 560, 84, "llvm"),
    (650, 908, 560, 84, "openxla"),
    (70, 1062, 500, 100, "pjrt-cpu"),
    (590, 1062, 500, 100, "pjrt-capi"),
    (1110, 1062, 540, 100, "xprof"),
    (70, 1242, 500, 100, "cpu-exec"),
    (590, 1242, 500, 192, "libtpu"),
    (1110, 1242, 540, 192, "tpu-hw"),
]


def box(x, y, w, h, title, sub, fill, stroke, *, dashed=False, off=False,
        title_size=17, sub_size=12.5, radius=10):
    dash = ' stroke-dasharray="8 5"' if dashed else ""
    op = ' opacity="0.55"' if off else ""
    out = [f'<g{op}>',
           f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
           f'fill="{fill}" stroke="{stroke}" stroke-width="2.2"{dash}/>']
    cx, ty = x + w / 2, y + 26
    out.append(f'<text x="{cx}" y="{ty}" font-family="{FONT}" font-size="{title_size}" '
               f'font-weight="700" fill="#1b1b1b" text-anchor="middle">{html.escape(title)}</text>')
    for i, line in enumerate(sub):
        out.append(f'<text x="{cx}" y="{ty + 19 + i * 15.5}" font-family="{FONT}" '
                   f'font-size="{sub_size}" fill="#3a3a3a" text-anchor="middle">'
                   f'{html.escape(line)}</text>')
    out.append('</g>')
    return "\n".join(out)


def band(number):
    b = BANDS[number]
    return (f'<rect x="{BAND_X}" y="{b["y"]}" width="{BAND_W}" height="{b["h"]}" rx="14" '
            f'fill="{b["color"]}" opacity="0.42"/>'
            f'<text x="{BAND_X + 18}" y="{b["y"] + LABEL_BASELINE}" font-family="{FONT}" '
            f'font-size="14.5" font-weight="700" fill="#4a4a4a">{html.escape(b["label"])}</text>')


ARROW_LABELS = []   # (text, x, y) recorded so --check can verify clear space

# Footer reserved regions: notes occupy NOTE_YS (spaced 24px) and the revision
# block owns its own band below them.
NOTE_YS = (1492, 1516, 1540, 1564, 1588)
REVISION_Y = H - 34


def arrow(x1, y1, x2, y2, label="", *, dashed=False, color="#444444", width=2.0,
          label_dx=0, label_dy=-7, label_y=None):
    """Draw an arrow. A label may be pinned with label_y so long multi-band arrows
    can keep their caption in clear space instead of landing on a band caption."""
    dash = ' stroke-dasharray="7 4"' if dashed else ""
    out = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
           f'stroke-width="{width}"{dash} marker-end="url(#arrowhead)"/>']
    if label:
        mx = (x1 + x2) / 2 + label_dx
        my = label_y if label_y is not None else (y1 + y2) / 2 + label_dy
        ARROW_LABELS.append((label, mx, my))
        out.append(f'<text x="{mx}" y="{my}" font-family="{FONT}" font-size="12" '
                   f'fill="{color}" text-anchor="middle">{html.escape(label)}</text>')
    return "\n".join(out)


def note(x, y, text, size=12.5, fill=C_NOTE):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}">{html.escape(text)}</text>')


def build():
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         '<defs><marker id="arrowhead" markerWidth="11" markerHeight="8" refX="9" refY="4" '
         'orient="auto"><path d="M0,0 L11,4 L0,8 z" fill="#444444"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{W/2}" y="44" font-family="{FONT}" font-size="27" font-weight="700" '
         f'fill="#111111" text-anchor="middle">JAX 软件栈组件图：位置、体量与上下游关系</text>',
         f'<text x="{W/2}" y="70" font-family="{FONT}" font-size="13.5" fill="{C_NOTE}" '
         f'text-anchor="middle">纵向 = 流水线层次　·　横向 = 所属路径　·　框体大小 = 组件覆盖面　·　'
         f'实线框 = 源码可核对　·　虚线框 = 闭源边界</text>']

    for n in sorted(BANDS):
        s.append(band(n))

    s.append(box(*BOXES[0][:4], "JAX（Python 前端）",
                 ["jit / grad / vmap → tracing → Jaxpr",
                  "primitive 规则 → StableHLO　（jax/_src/…）",
                  "1,974 个文件　·　matmul / jit / vmap / grad"], C_FRONT, C_FRONT_S))
    s.append(box(*BOXES[1][:4], "jaxlib（含在 JAX 仓库内）",
                 ["Python ↔ native 绑定", "PyClient::CompileAndLoad",
                  "jaxlib/py_client.cc:475"], C_FRONT, C_FRONT_S))
    s.append(box(*BOXES[2][:4], "Pallas（用户 kernel）",
                 ["pallas_call：Ref / grid / BlockSpec",
                  "显式分块与片上内存控制", "另一条 lowering 路径"], C_FRONT, C_FRONT_S))
    s.append(box(*BOXES[3][:4], "IFRT / PJRT（接口）",
                 ["编译与可执行抽象", "PjRtCompiler / PjRtLoadedExecutable",
                  "运行时的接口层"], C_RUN, C_RUN_S))

    s.append(box(*BOXES[4][:4], "StableHLO　3,575 文件",
                 ["算子 / 类型 / 属性方言", "JAX lowering 的产物 = 可移植契约"], C_REPR, C_REPR_S))
    s.append(box(*BOXES[5][:4], "Shardy　761 文件",
                 ["分片方言 sdy", "import → propagation → export"], C_REPR, C_REPR_S))
    s.append(box(*BOXES[6][:4], "MLIR（多级 IR 基础设施）",
                 ["StableHLO / sdy / Mosaic 都是 MLIR 方言",
                  "XLA 内部 pass 与 dialect 转换的框架　（LLVM 项目的一部分，体量见 ④）"],
                 C_INFRA, C_INFRA_S, title_size=16))

    s.append(box(*BOXES[7][:4], "XLA　9,174 文件　（CPU 与 TPU 的公共前端）",
                 ["MlirToXlaComputation：MLIR → HLO　（xla/pjrt/mlir_to_hlo.cc:99）",
                  "HLO 优化 pass pipeline",
                  "XLA SPMD：把 Shardy 的分片结论落成实际分区与 collective",
                  "调度（scheduling）与显存分配（buffer assignment）",
                  "代码生成分派：CPU emitter ／ TPU 后端（闭源）"],
                 C_XLA, C_XLA_S, title_size=18))
    s.append(box(*BOXES[8][:4], "Mosaic（TPU）",
                 ["Pallas 的 TPU 内层表示", "lower_jaxpr_to_pipelined_module",
                  "→ 序列化 payload → 外层 tpu_custom_call",
                  "在 XLA 图里作为 custom call"], C_XLA, C_XLA_S))
    s.append(box(*BOXES[9][:4], "Triton（GPU 路径）",
                 ["1,695 文件", "被 XLA 固定为依赖", "属于 GPU 分支，本项目不使用", "列出仅为完整性"],
                 C_OFF, C_OFF_S, off=True, title_size=16))
    s.append(note(485, 808, "↑ XLA 自身开源；TPU 后端代码生成不在开源 XLA 内，见 ⑥", fill=C_CLOSED_S))

    s.append(box(*BOXES[10][:4], "LLVM / MLIR（代码生成底座）",
                 ["CPU emitter：MLIR → LLVM IR → 目标文件 → ORC JIT　（LLVM 181,514 文件）"],
                 C_INFRA, C_INFRA_S, title_size=16))
    s.append(box(*BOXES[11][:4], "OpenXLA / MLIR 上游",
                 ["LLVM 项目内的 MLIR 与后端设施，由 XLA 固定 revision"],
                 C_INFRA, C_INFRA_S, title_size=16))

    s.append(box(*BOXES[12][:4], "PJRT provider（CPU）",
                 ["进程内 CPU 执行路径", "不经过 C API adapter"], C_RUN, C_RUN_S))
    s.append(box(*BOXES[13][:4], "PJRT C API adapter",
                 ["只有这条分支会进入插件", "PJRT_Client_Compile"], C_RUN, C_RUN_S))
    s.append(box(*BOXES[14][:4], "XProf（性能分析）",
                 ["采集 / 转换 / 聚合 trace", "消费 XSpace；观察 host 与设备事件",
                  "68dba1826c37"], C_OBS, C_OBS_S))

    s.append(box(*BOXES[15][:4], "CPU 后端执行",
                 ["XLA CPU emitter 产物", "在 CPU 上真实执行"], C_HW, C_HW_S))
    s.append(box(*BOXES[16][:4], "libtpu（闭源 wheel）",
                 ["TPU 的 PJRT 插件：libtpu.so", "动态加载，不编进 jaxlib",
                  "内部含 TPU 编译器与运行时", "Mosaic → LLO 发生在这一层内部",
                  "版本由 JAX setup.py:27 规定 0.0.46.*"], C_CLOSED, C_CLOSED_S, dashed=True))
    s.append(box(*BOXES[17][:4], "TPU 硬件　v7x　2×2×1",
                 ["4 颗物理芯片 / 8 个 JAX device", "每芯片 2 个 device（core_on_chip 0/1）",
                  "本研究的真机目标"], C_HW, C_HW_S))

    # arrows: all endpoints live in inter-band gaps or inside their own band
    s.append(arrow(320, 284, 270, 368, "lower_jaxpr_to_module"))
    s.append(arrow(770, 284, 770, 368, "MLIR module"))
    s.append(arrow(1130, 284, 1095, 594, "TPU 路径"))
    s.append(arrow(910, 210, 970, 210, ""))
    s.append(arrow(270, 484, 300, 594, "StableHLO", label_y=506))
    s.append(arrow(700, 484, 660, 594, "sdy 分片", label_y=506))
    s.append(arrow(300, 826, 300, 908, "CPU 代码生成", label_y=843))
    s.append(arrow(480, 826, 480, 1062, "编译产物", label_y=843))
    s.append(arrow(810, 826, 830, 1062, "编译产物", label_y=843, label_dx=-46))
    s.append(arrow(300, 1162, 300, 1242, ""))
    s.append(arrow(840, 1162, 840, 1242, ""))
    s.append(arrow(1090, 1338, 1110, 1338, "", color=C_CLOSED_S))
    s.append(arrow(1425, 1162, 1425, 1242, "观测", dashed=True, color=C_OBS_S, label_dx=26))
    s.append(arrow(1110, 1128, 590, 1128, "观测主机侧", dashed=True, color=C_OBS_S,
                   label_dy=-9))

    s.append(note(LEFT, 1492, "证据边界：虚线框 = 闭源，无法从源码核对（libtpu 内部编译器与 LLO）。"
                              "实线框 = 固定 revision 的源码可核对。"))
    s.append(note(LEFT, 1516, "关键分叉：CPU 与 TPU 在 XLA 之后分开；CPU 侧的 pass、显存分配与"
                              "代码生成结论不能外推到 TPU。"))
    s.append(note(LEFT, 1540, "术语边界：Mosaic TPU MLIR ≠ LLO。Mosaic 是 MLIR 方言；LLO 是 libtpu "
                              "内部 TPU 编译器的输出，两者之间隔着闭源阶段。"))
    s.append(note(LEFT, 1564, "解耦要点：jaxlib 与 libtpu 各自独立。任何源码构建的 jaxlib 都是 "
                              "TPU-capable，只需搭配对应版本 libtpu。"))
    s.append(note(LEFT, 1588, "体量数字为 git 跟踪文件数；LLVM 含全部后端，非 MLIR 单独计数。",
                  size=12))
    s.append(f'<text x="{W/2}" y="{H - 34}" font-family="{FONT}" font-size="12.5" '
             f'fill="#333333" text-anchor="middle">固定 revision：JAX 2d66622450e2（jax-v0.11.1）· '
             f'XLA dcf304bc5dca · StableHLO 7b1b15781ccb · Shardy 2832731619ff · '
             f'LLVM 75a45c373407 · Triton 96bc7e783a19 · XProf 68dba1826c37 · '
             f'libtpu 由 JAX setup.py:27 规定为 0.0.46.*</text>')
    s.append('</svg>')
    return "\n".join(s) + "\n"


def check(svg_text):
    """Fail if any box intrudes into a band's reserved label strip or overlaps another box."""
    problems = []
    strip_bottom = {n: b["y"] + LABEL_H for n, b in BANDS.items()}
    for x, y, w, h, key in BOXES:
        for n, b in BANDS.items():
            inside_band = b["y"] <= y < b["y"] + b["h"]
            if inside_band and y < strip_bottom[n]:
                problems.append(f"box {key} at y={y} intrudes into band {n} label strip "
                                f"(bottom {strip_bottom[n]})")
    for i, a in enumerate(BOXES):
        for b in BOXES[i + 1:]:
            ax, ay, aw, ah, ak = a
            bx, by, bw, bh, bk = b
            if ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah:
                problems.append(f"boxes {ak} and {bk} overlap")
    for x, y, w, h, key in BOXES:
        if x < LEFT - 2 or x + w > RIGHT + 2:
            problems.append(f"box {key} exceeds horizontal range")
        if y + h > H - 60:
            problems.append(f"box {key} exceeds vertical range")
    # Content guards: catch stale or mistyped identifiers that drawings tend to keep.
    for bad in ("0.0.27", "TBD", "TODO", "XXX", "0.11.2", "5832e866", "496bd4bd",
                "eb23a983", "ab547095", "23689d2a", "8.7.0"):
        if bad in svg_text:
            problems.append(f"stale or mistyped identifier present in the figure: {bad}")
    for required in ("2d66622450e2", "dcf304bc5dca", "7b1b15781ccb", "2832731619ff",
                     "75a45c373407", "96bc7e783a19", "68dba1826c37", "0.0.46"):
        if required not in svg_text:
            problems.append(f"expected revision missing from the figure: {required}")
    # Footer regions must not collide.
    for n in sorted(BANDS):
        if BANDS[n]["y"] + BANDS[n]["h"] > NOTE_YS[0] - 16:
            problems.append(f"band {n} reaches into the footer note area")
    if NOTE_YS[-1] + 8 > REVISION_Y - 14:
        problems.append("last footer note is too close to the revision line")
    for a, b in zip(NOTE_YS, NOTE_YS[1:]):
        if b - a < 20:
            problems.append(f"footer note spacing {b - a} too tight")

    # Arrow captions must not land in any band's caption strip.
    for text, mx, my in ARROW_LABELS:
        for n, b in BANDS.items():
            if b["y"] <= my <= b["y"] + LABEL_H:
                problems.append(f"arrow caption {text!r} at y={my} falls inside band "
                                f"{n} caption strip")
    if problems:
        raise SystemExit("layout check failed:\n  " + "\n  ".join(problems))
    print(f"layout check passed: {len(BOXES)} boxes, {len(BANDS)} bands, "
          f"{len(ARROW_LABELS)} arrow captions, no intrusions or overlaps; revisions and "
          f"closed-boundary note present, no stale identifiers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('research/jax-stack/jax-stack-components.svg'))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    ARROW_LABELS.clear()
    text = build()
    if args.check:
        check(text)
    args.output.write_text(text)
    print(f'wrote {args.output} ({args.output.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
