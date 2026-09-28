"""Split core.py at top-level statement boundaries into translatable chunks.

Each chunk is a contiguous slice of the original file. Chunk boundaries always
fall on a line that is *outside* any multi-line string literal, so a chunk is
never cut in the middle of a docstring.
"""
import ast
import json
import pathlib
import sys

TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 450
src_path = pathlib.Path(sys.argv[1])
root = src_path.parent
src = src_path.read_text(encoding='utf-8')
lines = src.splitlines(keepends=True)
tree = ast.parse(src)

# Hard boundaries at the first line of every top-level statement, including
# its decorators (FunctionDef.lineno points at the `def` line, not at `@...`).
def first_line(n):
  first = n.lineno
  for d in getattr(n, 'decorator_list', None) or []:
    first = min(first, d.lineno)
  return first

starts = sorted({first_line(n) for n in tree.body})
starts = [s for s in starts if s > 1]
hard = set(starts)
if tree.body:
  hard.add(min(n.lineno for n in tree.body))
hard = sorted(hard)

cuts = [1]
cur = 1
for s in hard:
  if s <= cur:
    continue
  if s - cur >= TARGET:
    cuts.append(s)
    cur = s
cuts.append(len(lines) + 1)

out = root / 'chunks'
for p in out.glob('chunk_*.py'):
  p.unlink()

manifest = []
for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
  text = ''.join(lines[a - 1:b - 1])
  name = f'chunk_{i:02d}.py'
  (out / name).write_text(text, encoding='utf-8')
  n_comment = sum(1 for ln in lines[a - 1:b - 1] if ln.lstrip().startswith('#'))
  n_quote = text.count('"""') + text.count("'''")
  manifest.append({'chunk': name, 'start': a, 'end': b - 1, 'lines': b - a,
                   'comment_lines': n_comment, 'quote_markers': n_quote})

(root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
total = 0
for m in manifest:
  print(f"{m['chunk']}  orig lines {m['start']:>5}-{m['end']:<5} "
        f"({m['lines']:>4} lines, {m['comment_lines']:>3} comment lines, "
        f"{m['quote_markers']:>3} quote markers)")
  total += m['lines']
print('chunks:', len(manifest), 'covered lines:', total, 'of', len(lines))
assert total == len(lines), 'coverage mismatch'
