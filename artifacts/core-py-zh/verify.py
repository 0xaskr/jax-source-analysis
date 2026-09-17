"""Verify a translated core.py against the original.

Checks:
  1. both files parse with ast
  2. the significant token stream (everything except # comments) is identical,
     with docstring string literals normalised to a placeholder
  3. every string literal that is NOT a docstring is byte-identical
  4. reports comment lines that still contain no CJK characters
"""
import ast
import io
import re
import sys
import tokenize

CJK = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')


def docstring_positions(src):
  """Return the full source span of every documentation string.

  This covers both real docstrings (the first statement of a module, class or
  function) and PEP 258 style documentation strings, i.e. any bare string
  expression statement such as the one that documents a module-level alias.
  """
  tree = ast.parse(src)
  pos = set()
  for node in ast.walk(tree):
    if (isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)):
      e = node
      pos.add((e.lineno, e.col_offset, e.end_lineno, e.end_col_offset))
  return pos


def in_span(tok, spans):
  for (l0, c0, l1, c1) in spans:
    if (tok.start[0], tok.start[1]) >= (l0, c0) and (tok.end[0], tok.end[1]) <= (l1, c1):
      return True
  return False


def significant_tokens(src):
  docs = docstring_positions(src)
  toks = []
  for tok in tokenize.generate_tokens(io.StringIO(src).readline):
    if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING):
      continue
    if tok.type == tokenize.STRING and in_span(tok, docs):
      toks.append((tok.type, '<DOCSTRING>'))
    else:
      toks.append((tok.type, tok.string))
  return toks


def comment_lines(src):
  out = []
  for tok in tokenize.generate_tokens(io.StringIO(src).readline):
    if tok.type == tokenize.COMMENT:
      out.append((tok.start[0], tok.string))
  return out


orig_path, new_path = sys.argv[1], sys.argv[2]
orig = open(orig_path, encoding='utf-8').read()
new = open(new_path, encoding='utf-8').read()

ok = True
for label, s in (('original', orig), ('translated', new)):
  try:
    ast.parse(s)
    print(f'[ok]   {label} parses')
  except SyntaxError as e:
    ok = False
    print(f'[FAIL] {label} SyntaxError: {e}')

a, b = significant_tokens(orig), significant_tokens(new)
if a == b:
  print(f'[ok]   significant token streams identical ({len(a)} tokens)')
else:
  ok = False
  print(f'[FAIL] token streams differ: orig={len(a)} translated={len(b)}')
  for i, (x, y) in enumerate(zip(a, b)):
    if x != y:
      print(f'       first difference at significant token #{i}')
      print(f'       original  : {x}')
      print(f'       translated: {y}')
      print('       context original  :', [t[1] for t in a[max(0, i - 6):i + 6]])
      print('       context translated:', [t[1] for t in b[max(0, i - 6):i + 6]])
      break

co, cn = comment_lines(orig), comment_lines(new)
print(f'[info] comment count: original={len(co)} translated={len(cn)}')
if len(cn) > len(co):
  ok = False
  print('[FAIL] comment count grew (comments were added)')
elif len(co) - len(cn) > 0:
  print(f'[warn] {len(co) - len(cn)} comment line(s) fewer than the original '
        f'(allowed only where a multi-line comment was re-flowed)')

english = [(ln, txt) for ln, txt in cn if not CJK.search(txt)
           and re.search(r'[A-Za-z]{3,}', txt)]
print(f'[info] translated comments without CJK characters: {len(english)}')
for ln, txt in english[:40]:
  print(f'       line {ln}: {txt[:100]}')

print('RESULT:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
