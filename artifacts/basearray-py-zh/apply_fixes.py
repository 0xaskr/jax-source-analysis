"""Apply the review fixes to the assembled Chinese basearray.py."""
import pathlib
import sys

p = pathlib.Path(__file__).parent / 'basearray.py'
src = p.read_text(encoding='utf-8')

FIXES = [
  ("# TODO(jakevdp): 修复导入环并定义这些类型。",
   "# TODO(jakevdp): 修复循环导入并定义这些类型。"),
  ("    isinstance(x, jax.Array)  # 在追踪函数内外都返回 True。",
   "    isinstance(x, jax.Array)  # 在被追踪函数内部和外部都返回 True。"),
  ("  不应用 ``jax.Array`` 直接创建数组；而应使用 :mod:`jax.numpy` 提供的",
   "  不应直接使用 ``jax.Array`` 来创建数组；而应使用 :mod:`jax.numpy` 提供的"),
  ("  # 为了静态类型分析，这些定义在配套的 basearray.pyi 文件中有一份镜像。",
   "  # 为了静态类型分析，这些定义在配套的 basearray.pyi 文件中有对应的镜像定义。"),
  ('    """返回特定索引处可寻址数据的数组。"""',
   '    """返回特定索引处的可寻址数据所组成的数组。"""'),
  ("    如果当前进程能够访问 :class:`Sharding` 中列出的所有设备，那么该",
   "    如果当前进程能够寻址到 :class:`Sharding` 中指定的所有设备，那么该"),
  ("    jax.Array 可能横跨多台主机，因而不是完全可寻址的。",
   "    jax.Array 可能横跨多台主机，并且不是完全可寻址的。"),
  ("    同一（些）设备上。对提交到不同设备的参数调用同一个运算则会报错。",
   "    同一（些）设备上。对已提交到不同设备上的参数调用运算则会报错。"),
  ("    该数组的值。通常，当用户请求读取设备上数组的值时，这件事会在背后\n"
   "    发生；但如果要一直等到用户请求时才发起设备到主机的复制，JAX 就必须",
   "    该数组的值。通常，当用户请求读取设备上数组的值时，这会在幕后自动\n"
   "    发生；但如果要一直等到用户请求时才发起设备到主机的复制，JAX 就必须"),
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
