"""Concatenate translated chunks and verify the result against the original."""
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).parent
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
orig = root / 'core.py.orig'

parts = []
missing = []
bad_newlines = []
orig_lines_all = orig.read_text(encoding='utf-8').splitlines(keepends=True)
for m in manifest:
  p = root / 'out' / m['chunk']
  if not p.exists():
    missing.append(m['chunk'])
    continue
  text = p.read_text(encoding='utf-8')
  lines = text.splitlines(keepends=True)
  orig_slice = orig_lines_all[m['start'] - 1:m['end']]
  orig_lines = m['lines']
  flags = []
  if len(lines) != orig_lines:
    flags.append('LINES')
  elif [ln.endswith('\n') for ln in lines] != [ln.endswith('\n') for ln in orig_slice]:
    flags.append('NEWLINE')
    bad_newlines.append(m['chunk'])
  if lines and not lines[-1].endswith('\n'):
    flags.append('NO-FINAL-NEWLINE')
    bad_newlines.append(m['chunk'])
  print(f"{'ok ' if not flags else ','.join(flags)} {m['chunk']}: original {orig_lines} lines, "
        f"translated {len(lines)} lines (orig lines {m['start']}-{m['end']})")
  parts.append(text)

if missing:
  print('MISSING CHUNKS:', ', '.join(missing))
  sys.exit(2)
if bad_newlines:
  print('BAD TRAILING NEWLINES:', ', '.join(bad_newlines))
  sys.exit(3)

final = root / 'core.py'
final.write_text(''.join(parts), encoding='utf-8')
print(f'\nassembled -> {final} ({len(final.read_text(encoding="utf-8").splitlines())} lines)')

r = subprocess.run([sys.executable, str(root / 'verify.py'), str(orig), str(final)],
                   capture_output=True, text=True)
tail = '\n'.join(r.stdout.splitlines()[:12])
print(tail)
print('verify exit code:', r.returncode)
sys.exit(r.returncode)
