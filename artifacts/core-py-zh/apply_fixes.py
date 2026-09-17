"""Apply the review fixes to the assembled Chinese core.py.

Every replacement is an exact-match, single-occurrence edit; the script aborts
without writing if any pattern is missing or ambiguous.
"""
import pathlib
import sys

p = pathlib.Path(__file__).parent / 'core.py'
src = p.read_text(encoding='utf-8')

FIXES = [
  # --- reviewer 1: original lines 1-1107 --------------------------------
  # redundant duplicate half-sentence (orig 184-186)
  ("      # 我们不得不允许在 debug_info 缺失时进行调用。\n"
   "      # 缺失 debug_info 时我们也必须允许调用。\n",
   "      # 必须允许 debug_info 缺失时的调用。\n"),
  # trace terminology (orig 849, 854-856)
  ("    # 我们经常需要某个 trace 的弱引用，所以预先计算一个。",
   "    # 我们经常需要某个追踪（Trace）的弱引用，所以预先计算一个。"),
  ('    """把一个值提升到某个 trace 中。',
   '    """把一个值提升到某个追踪中。'),
  # phrase split across lines (orig 352-357)
  ("# 这个上下文管理器相当热，因为它经常针对每个 jaxpr 方程\n"
   "# 被调用。\n",
   "# 这个上下文管理器处于热点路径，因为它会针对每个 jaxpr 方程被频繁调用。\n"),
  # phrase split across lines (orig 718-720)
  ("    # 这等价于 \"with take_current_trace()\"，但 bind() 代码\n"
   "    # 被频繁调用，避免使用上下文\n"
   "    # 管理器对象会略快一些。\n",
   "    # 这等价于 \"with take_current_trace()\"，但 bind() 代码\n"
   "    # 被频繁调用，避免使用上下文管理器对象会略快一些。\n"),

  # --- reviewer 2: original lines 1108-2214 -----------------------------
  # "obvious" was dropped (orig 2055)
  ("# 在编译器中没有直接对应的基本类型（“物理”）数组。特别地，它们的元素类型",
   "# 在编译器中并没有显而易见的直接对应的基本类型（“物理”）数组。特别地，"
   "其元素类型"),
  # "hack" left dangling on its own line (orig 1319-1327)
  ("  # 并采用一种只缓存顶层函数的更简单的缓存方案。那时我们就可以\n"
   "  # 移除这个\n"
   "  # hack。\n",
   "  # 并采用一种只缓存顶层函数的更简单的缓存方案。那时我们就可以\n"
   "  # 移除这个 hack。\n"),
  # terminology (orig 1218)
  ("  # 仅对已具现化(materialized)数组有效的方法",
   "  # 仅对已实体化(materialized)数组有效的方法"),
  # wording (orig 2102)
  ("  # 维度绝大多数情况下是整数（远远如此），所以我们先检查这一情况。",
   "  # 维度绝大多数情况下是整数（且远超其他情况），所以我们先检查这一情况。"),
  # wording (orig 1616)
  ("  for tracer in tracers:  # 不用生成器表达式：它会被 gc 看到并自我报告",
   "  for tracer in tracers:  # 不用生成器表达式：它会被 gc 看到，并把自身也报告为引用者"),

  # --- reviewer 3: original lines 2215-3321 -----------------------------
  # half-translated "shaped array" (orig 3060)
  ("# 当需要形状/数据类型时，所有抽象 token 使用的单例 shaped array。",
   "# 当需要形状/数据类型时，所有抽象 token 共用的单例 ShapedArray。"),
  # "穿入穿出" is not idiomatic (orig 3066-3067)
  ("  # token 包装的底层数据，可用于在计算中\n"
   "  # 穿入穿出以构建数据依赖。\n",
   "  # token 包装的底层数据，可以传入和传出计算，从而构建数据依赖。\n"),
  # mutation != mutable; keep `ref` as the code term (orig 2902-2904)
  ("      可变语义。\n"
   "    pin: 是否在 HLO 中将该引用降级为 pinned 缓冲区。\n",
   "      变更语义。\n"
   "    pin: 是否在 HLO 中把该 ref 降级为 pinned 缓冲区。\n"),
  # redundant gloss (orig 2391)
  ("  # 没有 __eq__ 或 __hash__：驻留类使用对象标识（identity）。",
   "  # 没有 __eq__ 或 __hash__：驻留类使用对象标识。"),
  # wording (orig 2559)
  ("  # 与 numpy 的错误相同", "  # 与 numpy 报错相同"),

  # --- reviewer 4: original lines 3322-4428 -----------------------------
  # untranslated section divider (orig 4043)
  ("# ------------------- Jaxpr printed representation -------------------",
   "# ------------------- Jaxpr 打印表示 -------------------"),
  # redundant tail (orig 3465-3469)
  ("# 使用，用户代码永远拿不到对该对象的引用。我们不想使用函数\n"
   "# 对象本身，因为那可能会持久保留对函数对象的引用，我们\n"
   "# 不希望这样。\n",
   "# 使用，用户代码永远拿不到对该对象的引用。我们不想使用函数\n"
   "# 对象本身，因为那可能会持久保留对函数对象的引用。\n"),
  # misreadable (orig 3606)
  ("  - 变量在整个 jaxpr 中具有相同的类型",
   "  - 同一变量在整个 jaxpr 中类型保持一致"),
  # "emit" / "identity" (orig 3388)
  ("# 只发射一个方程，该方程在重新追踪下保持自身的同一性。它的",
   "# 只生成一个方程，该方程在重新追踪时保持不变。它的"),
  # aliasing direction (orig 4140)
  ("    `for_vars` 是互不相同的 Var，并且与 `like_vars` 互为别名。",
   "    `for_vars` 是互不相同的 Var，并被作为 `like_vars` 的别名。"),
  # wording (orig 3749)
  ("          # 原语解除命名轴效果是合法的。",
   "          # 原语可以合法地消解（discharge）命名轴效果。"),
  # wording (orig 3578)
  ("    # TODO(yashkatariya,mattjj): 添加对 types 中分片不匹配的检查",
   "    # TODO(yashkatariya,mattjj): 添加对类型中分片（sharding-in-types）不匹配的检查"),
  # wording (orig 3858)
  ("    dtype: 一个类 dtype 对象", "    dtype: 一个类似 dtype 的对象"),
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
