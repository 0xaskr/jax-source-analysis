#!/usr/bin/env python3
"""Render a human-readable digest of source-index.json.

The index is a machine-readable JSON artifact; the kickoff's second deliverable
also asks for a Markdown guide. This generator keeps the digest derived from the
index rather than hand-maintained, so the two cannot drift.

Usage:
  .venv/bin/python -B research/jax-stack/render_source_index.py \
    --output research/jax-stack/source-index-digest.md
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / 'research/jax-stack/source-index.json'

COMPONENT_ORDER = ['jax', 'stablehlo', 'shardy', 'xla', 'llvm', 'xprof', 'unknown']

COMPONENT_ROLE = {
    'jax': 'Python 前端：API、tracing、Jaxpr、primitive 规则与 MLIR lowering',
    'stablehlo': '可移植方言定义：操作、类型与属性',
    'shardy': '分片方言与 import/propagation/export pass 组织',
    'xla': 'MLIR/HLO 转换、后端优化、调度、buffer assignment 与代码生成',
    'llvm': 'CPU 代码生成：LLVM IR、MC、ORC JIT',
    'xprof': '性能数据采集、转换与聚合 tooling',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'research/jax-stack/source-index-digest.md')
    parser.add_argument('--limit-per-layer', type=int, default=200)
    args = parser.parse_args()

    data = json.loads(INDEX.read_text())
    entries = data['entries']
    edges = data['edges']

    by_component = defaultdict(list)
    for e in entries:
        by_component[e.get('component', 'unknown')].append(e)
    for v in by_component.values():
        v.sort(key=lambda e: (e.get('layer', ''), e.get('id', '')))

    edge_out = Counter()
    edge_in = Counter()
    for e in edges:
        edge_out[e.get('caller')] += 1
        edge_in[e.get('callee')] += 1

    lines = []
    add = lines.append
    add('# 源码与 API 索引导读（由 source-index.json 生成）')
    add('')
    add('本文件由 `render_source_index.py` 从 [source-index.json](source-index.json) 生成，')
    add('不是手工维护的副本；两者不一致时应重新生成而不是手改。')
    add('')
    add(f"- kickoff revision: {data.get('kickoff_revision')}")
    add(f"- schema_version: {data.get('schema_version')}")
    add(f"- 入口总数: **{len(entries)}**，关系总数: **{len(edges)}**")
    add('')
    # The index records the revisions that were indexed. Compare per root, not as
    # a set: revisions shared by more than one root (and roots this script does
    # not re-read, such as xprof) would otherwise hide a genuine pin change.
    drifted = []
    for name, root in sorted((data.get('source_roots') or {}).items()):
        if not isinstance(root, dict):
            continue
        rel, indexed = root.get('path'), root.get('revision')
        try:
            live = subprocess.run(['git', '-C', str(ROOT / rel), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            continue
        if indexed and live and indexed != live:
            drifted.append((name, rel, indexed, live))
    if drifted:
        add('> **注意：本索引记录的是换 pin 之前的源码 revision。**')
        add('> 工作区已换到 stable 0.11.1（见 [交接文档 4.1](AGENT-HANDOFF-2026-09-15.md)），')
        add('> 下列组件已不一致，索引的 `revision` 与 `source_sha256` 需针对新 pin 重新核对：')
        add('>')
        for name, rel, indexed, live in drifted:
            add(f'> - `{name}`（`{rel}`）：索引 `{indexed[:12]}` → 现为 `{live[:12]}`')
        add('')
    add('## 组件覆盖')
    add('')
    add('| 组件 | 入口数 | 关系（出/入） | 职责 |')
    add('|---|---|---|---|')
    for c in COMPONENT_ORDER + [k for k in by_component if k not in COMPONENT_ORDER]:
        if c not in by_component:
            continue
        n_out = sum(edge_out[e['id']] for e in by_component[c])
        n_in = sum(edge_in[e['id']] for e in by_component[c])
        add(f"| `{c}` | {len(by_component[c])} | {n_out}/{n_in} | {COMPONENT_ROLE.get(c, '—')} |")
    add('')
    add('各固定 revision：')
    add('')
    add('| 组件 | revision |')
    add('|---|---|')
    for c, root in sorted((data.get('source_roots') or {}).items()):
        rev = root.get('revision') if isinstance(root, dict) else root
        add(f"| `{c}` | `{rev}` |")
    add('')
    add('## 证据级别分布')
    add('')
    levels = Counter(e.get('evidence_level') for e in entries)
    add('| 证据级别 | 入口数 |')
    add('|---|---|')
    for k, v in levels.most_common():
        add(f'| `{k}` | {v} |')
    add('')
    add('> `SOURCE-ONLY` 表示只核对了固定源码，没有运行产物。索引条目的证据级别描述的是')
    add('> **该入口被核对的方式**，不是整个研究目标的完成状态。')
    add('')
    add('## 索引声明的边界')
    add('')
    for lim in data.get('limits', []):
        add(f'- {lim}')
    add('')
    add('## 入口清单（按组件与层次）')
    add('')
    for c in COMPONENT_ORDER + [k for k in by_component if k not in COMPONENT_ORDER]:
        if c not in by_component:
            continue
        add(f'### `{c}` — {len(by_component[c])} 个入口')
        add('')
        add('| 层次 | 符号 | 位置 | 输入 → 输出 | 约束 | 关系 |')
        add('|---|---|---|---|---|---|')
        shown = 0
        for e in by_component[c]:
            if shown >= args.limit_per_layer:
                add(f'| … | … | 其余 {len(by_component[c]) - shown} 项见 JSON | | | |')
                break
            loc = f"{e.get('path','')}#L{e.get('line')}"
            rel = []
            if e.get('callers'):
                rel.append('↑' + ', '.join(f'`{x}`' for x in e['callers'][:3]))
            if e.get('callees'):
                rel.append('↓' + ', '.join(f'`{x}`' for x in e['callees'][:3]))
            io = f"{e.get('inputs','') or '—'} → {e.get('outputs','') or '—'}"
            add(f"| {e.get('layer','')} | `{e.get('symbol','')}` | `{loc}` | "
                f"{io.replace('|', '/')} | {(e.get('constraint') or '—').replace('|', '/')} | "
                f"{' '.join(rel) or '—'} |")
            shown += 1
        add('')

    add('## 关系清单')
    add('')
    add('| caller | → callee | 种类 | 调用位置 | 条件 |')
    add('|---|---|---|---|---|')
    for e in edges:
        loc = f"{e.get('path','')}#L{e.get('line')}"
        add(f"| `{e.get('caller')}` | `{e.get('callee')}` | {e.get('kind','')} | `{loc}` | "
            f"{(e.get('condition') or '—').replace('|', '/')} |")
    add('')

    args.output.write_text('\n'.join(lines) + '\n')
    print(f'wrote {args.output} ({len(lines)} lines, {args.output.stat().st_size} bytes)')
    print('components:', {c: len(v) for c, v in by_component.items()})
    print('edges:', len(edges), 'evidence:', dict(levels))


if __name__ == '__main__':
    main()
