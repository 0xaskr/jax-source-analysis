"""Apply the review fixes to the assembled Chinese partial_eval.py."""
import pathlib
import re
import sys

p = pathlib.Path(__file__).parent / 'partial_eval.py'
src = p.read_text(encoding='utf-8')

FIXES = [
  # --- reviewer 2: original lines 811-1620 ------------------------------
  ("        # TODO(slebedev): 这是一个合理错误，需要修改 BUILD 才能解决。",
   "        # TODO(slebedev): 这是一个真实的（类型检查）报错，需要修改 BUILD 才能解决。"),
  ("#    'known' 一侧 jaxpr 的输出引出，并作为输入绑定添加到\n"
   "#    'unknown' 的 jaxpr）。\n",
   "#    'known' 一侧 jaxpr 的输出引出，并作为输入绑定变量（input binder）加入\n"
   "#    'unknown' 一侧的 jaxpr）。\n"),
  ("  # 通过移除前向传递来精简 jaxpr_known_ 的输出。",
   "  # 通过移除被转发的输出（forwards）来精简 jaxpr_known_ 的输出。"),
  ("  # 计算哪些输入只是被前向传递到输出。",
   "  # 计算哪些输入只是被转发（forward）到输出。"),
  # dangling half-sentence: the English sentence spans two comment lines
  ("  # 规则需要同时兼容两者。TODO(dougalm)：等我们修好转发后移除这一点。\n"
   "  # 转发。\n",
   "  # 规则需要同时兼容两者。TODO(dougalm): 等我们修好转发后，\n"
   "  # 就移除这一点。\n"),

  # --- reviewer 3: original lines 1621-2428 -----------------------------
  ("    severity 是一个整数，越小表示越接近。",
   "    severity 是一个整数，越小越好。"),
  ("    in_avals: ft.FlatTree,  # (args, kwargs) 二元组",
   "    in_avals: ft.FlatTree,  # (args, kwargs) 对"),
  ("dce_jaxpr_call_rule = dce_jaxpr_closed_call_rule  # 为下游用户保留的别名",
   "dce_jaxpr_call_rule = dce_jaxpr_closed_call_rule  # 供下游用户使用的别名"),
]

failures = []
for old, new in FIXES:
  n = src.count(old)
  if n != 1:
    failures.append((n, old.splitlines()[0][:80]))
    continue
  src = src.replace(old, new)

if failures:
  for n, head in failures:
    print(f'ABORT: pattern found {n}x: {head}')
  sys.exit(1)

# Unify the marker punctuation: `# TODO(x)：` -> `# TODO(x): ` (and TODO/FIXME/
# XXX/NOTE without a parenthesised name).  Only matches directly after a marker,
# so ordinary Chinese full-width colons elsewhere are untouched.
marker = re.compile(r'(#\s*(?:TODO|FIXME|XXX|NOTE)(?:\([^)\n]*\))?)\s*[：:]\s*')
src, k = marker.subn(r'\1: ', src)

p.write_text(src, encoding='utf-8')
print(f'applied {len(FIXES)} fixes + {k} marker-colon normalisations -> {p} '
      f'({len(src.splitlines())} lines)')
