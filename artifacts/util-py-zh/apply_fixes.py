"""Apply the review fixes to the assembled Chinese util.py."""
import pathlib
import sys

p = pathlib.Path(__file__).parent / 'util.py'
src = p.read_text(encoding='utf-8')

FIXES = [
  # --- reviewer 1: original lines 1-411 --------------------------------
  ("  # 中 builtins.zip 类似的策略。这支持最多三个参数时返回类型\n"
   "  # 与输入类型相匹配。\n",
   "  # 中 builtins.zip 类似的策略。这样在最多三个参数时，返回类型\n"
   "  # 可以与输入类型相匹配。\n"),
  ('  """将长度为 2 的元组序列解包为两个元组。"""',
   '  """将由长度为 2 的元组构成的序列解包为两个元组。"""'),
  ('  """将长度为 3 的元组序列解包为三个元组。"""',
   '  """将由长度为 3 的元组构成的序列解包为三个元组。"""'),
  ('  """在列表中替换取值。"""',
   '  """替换列表中的取值，并返回新的元组。"""'),
  ("    for_what: 用于标识此缓存用途的字符串。它\n"
   "       用于调试。\n",
   "    for_what: 用于标识此缓存用途的字符串。\n"
   "       该字段用于调试。\n"),
  ('  """用于避免不可变驻留类样板代码的装饰器。"""',
   '  """用于避免为不可变的驻留类编写样板代码的装饰器。"""'),
  ("  # 阻止构造之后发生修改。",
   "  # 抑制构造完成后的修改。"),

  # --- reviewer 2: original lines 412-816 ------------------------------
  ('  """将函数的相等性与哈希与其身份解耦。',
   '  """将函数的相等性和哈希与其身份解耦。'),
  ("  闭包中的相关元素（例如，当某些被闭包捕获的值\n"
   "  不可哈希，但却完全由可哈希的局部变量决定时，\n"
   "  只要依据 `closure` 中的内容就能推导出它们，\n"
   "  这种情况也是允许的）。\n",
   "  闭包中的相关元素（例如，当某些被闭包捕获的值\n"
   "  不可哈希，但完全由可哈希的局部变量决定时，\n"
   "  这种情况也是允许的）。\n"),
  ("# 一个方便的 (args, kwargs) 对容器，让你可以映射它等等，\n"
   "# 其 API 与 `FlatTree` 类似。\n",
   "# 一个便捷的 (args, kwargs) 对容器，可对其做 map 等操作，\n"
   "# API 与 `FlatTree` 类似。\n"),
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

p.write_text(src, encoding='utf-8')
print(f'applied {len(FIXES)} fixes -> {p} ({len(src.splitlines())} lines)')
