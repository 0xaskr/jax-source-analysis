class Unhashable:
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  def __eq__(self, other):
    return self.val == other.val

class Hashable:
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  def __hash__(self):
    return hash(self.val)

  def __eq__(self, other):
    return self.val == other.val

def wrap_name(transform_name: str, name: str) -> str:
  return f"{transform_name}({name})"


def fun_name(fun: Callable, default_name: str = "<unnamed function>") -> str:
  name = getattr(fun, "__name__", None)
  if name is not None:
    return name
  if isinstance(fun, partial):
    return fun_name(fun.func)
  else:
    return default_name


def fun_qual_name(fun: Callable) -> str:
  qual_name = getattr(fun, "__qualname__", None)
  if qual_name is not None:
    return qual_name
  if isinstance(fun, partial):
    return fun_qual_name(fun.func)
  return fun_name(fun)

def canonicalize_axis(axis: SupportsIndex, num_dims: int) -> int:
  """将 [-num_dims, num_dims) 范围内的轴规范化到 [0, num_dims)。"""
  axis = operator.index(axis)
  if not -num_dims <= axis < num_dims:
    raise ValueError(f"axis {axis} is out of bounds for array of dimension {num_dims}")
  if axis < 0:
    axis = axis + num_dims
  return axis

def canonicalize_axis_tuple(axis: int | Sequence[int] | None, ndim: int, allow_duplicate: bool = False) -> tuple[int, ...]:
  if axis is None:
    return tuple(range(ndim))
  if isinstance(axis, Sequence):
    axis = tuple(canonicalize_axis(i, ndim) for i in axis)
    if not allow_duplicate and len(set(axis)) != len(axis):
      raise ValueError(f"repeated axis: {axis}")
    return axis
  else:
    return (canonicalize_axis(axis, ndim),)

def moveaxis(x: Array, src: int | Sequence[int], dst: int | Sequence[int]) -> Array:
  if src == dst:
    return x
  if isinstance(src, int):
    src = (src,)
  if isinstance(dst, int):
    dst = (dst,)
  src = [canonicalize_axis(a, x.ndim) for a in src]
  dst = [canonicalize_axis(a, x.ndim) for a in dst]
  perm = [i for i in range(np.ndim(x)) if i not in src]
  for d, s in sorted(zip(dst, src)):
    perm.insert(d, s)
  return x.transpose(perm)

def ceil_of_ratio(x: int, y: int) -> int:
  return -(-x // y)


def wraps[T](
    wrapped: Callable,
    namestr: str | None = None,
    docstr: str | None = None,
    **kwargs,
) -> Callable[[T], T]:
  """
  类似 `functools.wraps`，但对结果函数的名称和文档字符串
  提供了更细粒度的控制。
  """
  def wrapper(fun: T) -> T:
    try:
      name = fun_name(wrapped)
      doc = getattr(wrapped, "__doc__", "") or ""
      fun.__dict__.update(getattr(wrapped, "__dict__", {}))
      fun.__annotations__ = getattr(wrapped, "__annotations__", {})
      fun.__name__ = name if namestr is None else namestr.format(fun=name)  # pyrefly: ignore[missing-attribute]
      fun.__module__ = getattr(wrapped, "__module__", "<unknown module>")
      fun.__doc__ = (doc if docstr is None
                     else docstr.format(fun=name, doc=doc, **kwargs))
      fun.__qualname__ = getattr(wrapped, "__qualname__", fun.__name__)  # pyrefly: ignore[missing-attribute]
      fun.__wrapped__ = wrapped  # pyrefly: ignore[missing-attribute]
    except Exception:
      pass
    return fun
  return wrapper

def tuple_insert[T](t: tuple[T, ...], idx: int, val: T) -> tuple[T, ...]:
  assert 0 <= idx <= len(t), (idx, len(t))
  return t[:idx] + (val,) + t[idx:]

def tuple_delete[T](t: tuple[T, ...], idx: int) -> tuple[T, ...]:
  assert 0 <= idx < len(t), (idx, len(t))
  return t[:idx] + t[idx + 1:]

def tuple_update[T](t: tuple[T, ...], idx: int, val: T) -> tuple[T, ...]:
  assert 0 <= idx < len(t), (idx, len(t))
  return t[:idx] + (val,) + t[idx+1:]

class HashableFunction:
  """将函数的相等性与哈希与其身份解耦。

  局部 lambda 和函数定义在每次函数调用时都会被重新分配，这使得在不同调用中
  创建出的函数比较起来并不相等。这会破坏我们的缓存逻辑，
  而该逻辑实际上只应关心比较语义，
  而不应关心真实的身份。

  该类使得可以基于语义来比较不同的函数。所考虑的部分包括：
  被包装函数的字节码（它由 CPython 解释器缓存，并且在
  外层函数的多次调用之间保持稳定），以及 `closure` 参数，
  该参数应包含作用域中所有会影响函数语义的值。特别是，
  `closure` 应当包含函数闭包的所有元素，或者说应当
  能够仅依据 `closure` 参数的内容，推导出真实函数
  闭包中的相关元素（例如，当某些被闭包捕获的值
  不可哈希，但却完全由可哈希的局部变量决定时，
  只要依据 `closure` 中的内容就能推导出它们，
  这种情况也是允许的）。
  """

  def __init__(self, f, closure):
    self.f = f
    self.closure = closure

  def __eq__(self, other):
    return (type(other) is HashableFunction and
            self.f.__code__ == other.f.__code__ and
            self.closure == other.closure)

  def __hash__(self):
    return hash((self.f.__code__, self.closure))

  def __call__(self, *args, **kwargs):
    return self.f(*args, **kwargs)

  def __repr__(self):
    return f'<hashable {self.f.__name__} with closure={self.closure}>'


class HashablePartial:
  def __init__(self, f, *args, **kwargs):
    self.f = f
    self.args = args
    self.kwargs = kwargs

  def __eq__(self, other):
    return (type(other) is HashablePartial and
            self.f.__code__ == other.f.__code__ and
            self.args == other.args and self.kwargs == other.kwargs)

  def __hash__(self):
    kwargs = tuple(sorted(self.kwargs.items(), key=lambda kv: kv[0]))
    return hash((self.f.__code__, self.args, kwargs))

  def __call__(self, *args, **kwargs):
    return self.f(*self.args, *args, **self.kwargs, **kwargs)

def maybe_named_axis(axis, if_pos, if_named):
  try:
    pos = operator.index(axis)
  except TypeError:
    return if_named(axis)
  else:
    return if_pos(pos)

def distributed_debug_log(*pairs):
  """如果启用了 `config.jax_distributed_debug`，则格式化并记录 `pairs`。

  Args:
    pairs: 要记录的标签/值对序列。第一对被当作后续各对的
    标题。
  """
  if config.distributed_debug.value:
    lines = ["\nDISTRIBUTED_DEBUG_BEGIN"]
    try:
      lines.append(f"{pairs[0][0]}: {pairs[0][1]}")
      for label, value in pairs[1:]:
        lines.append(f"  {label}: {value}")
    except Exception as e:
      lines.append("DISTRIBUTED_DEBUG logging failed!")
      lines.append(f"{e}")
    lines.append("DISTRIBUTED_DEBUG_END")
    logger.warning("\n".join(lines))


def stable_unique[T](it: Iterable[T]) -> Iterable[T]:
  """按出现顺序返回 `it` 中的唯一元素。

  这些元素必须是可哈希的。
  """
  return dict.fromkeys(it).keys()


class OrderedSet[ElementType]:
  elts_set: set[ElementType]
  elts_list: list[ElementType]

  def __init__(self):
    self.elts_set = set()
    self.elts_list = []

  def add(self, elt: ElementType) -> None:
    if elt not in self.elts_set:
      self.elts_set.add(elt)
      self.elts_list.append(elt)

  def update(self, elts: Seq[ElementType]) -> None:
    for e in elts:
      self.add(e)

  def __iter__(self) -> Iterator[ElementType]:
    return iter(self.elts_list)

  def __len__(self) -> int:
    return len(self.elts_list)

  def __contains__(self, elt: ElementType) -> bool:
    return elt in self.elts_set


class HashableWrapper:
  x: Any
  hash: int | None
  def __init__(self, x):
    self.x = x
    try: self.hash = hash(x)
    except: self.hash = None
  def __hash__(self):
    return self.hash if self.hash is not None else id(self.x)
  def __eq__(self, other):
    if not isinstance(other, HashableWrapper):
      return False
    return self.x == other.x if self.hash is not None else self.x is other.x

@dataclass(frozen=True)
class Either():
  is_right : bool
  val : Any

  def from_left(self):
    assert not self.is_right
    return self.val

  def from_right(self):
    assert self.is_right
    return self.val

  @property
  def is_left(self): return not self.is_right

  @staticmethod
  def left(x): return Either(False, x)

  @staticmethod
  def right(x): return Either(True, x)

  def __repr__(self):
    if self.is_left:
      return f"Left({self.val})"
    else:
      return f"Right({self.val})"

# 一个方便的 (args, kwargs) 对容器，让你可以映射它等等，
# 其 API 与 `FlatTree` 类似。
# TODO: 哈希、相等性、打印等
class PyArgs:
  def __init__(self, args, kwargs):
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    self.args = args
    self.kwargs = kwargs

  # True 表示保留
  def filter_with_mask(self, mask):
    assert len(mask) == len(self)
    keeps = iter(mask)
    return PyArgs(
        tuple(x for x in self.args if next(keeps)),
        {k: v for k, v in self.kwargs.items() if next(keeps)})

  @property
  def args_kwargs(self):
    return (self.args, self.kwargs)

  def map(self, f):
    return PyArgs(
        tuple(f(x) for x in self.args),
        {k: f(x) for k, x in self.kwargs.items()})

  def map2(self, ys, f):
    ys_iter = iter(ys)
    return self.map(lambda x: f(x, next(ys_iter)))

  def __len__(self):
    return len(self.args) + len(self.kwargs)

def _original_func(f: Callable) -> Callable:
  if isinstance(f, property):
    fget = cast(property, f).fget
    assert fget is not None
    return fget
  elif isinstance(f, functools.cached_property):
    return f.func
  return f


_T = TypeVar("_T")

def set_module(module: str) -> Callable[[_T], _T]:
  def wrapper(func: _T) -> _T:
    if module is not None:
      func.__module__ = module
    return func
  return wrapper


def use_cpp_class(cpp_cls: type[Any]) -> Callable[[type[_T]], type[_T]]:
  """一个在运行时用 C++ 版本替换 Python 类的装饰器。"""

  def wrapper(cls):
    if cpp_cls is None:
      return cls

    exclude_methods = {'__module__', '__dict__', '__doc__'}

    for attr_name, attr in cls.__dict__.items():
      if attr_name not in exclude_methods:
        if not hasattr(_original_func(attr), "_use_cpp"):
          setattr(cpp_cls, attr_name, attr)

    cpp_cls.__doc__ = cls.__doc__
    return cpp_cls

  return wrapper

def use_cpp_method(is_enabled: bool = True) -> Callable[[_T], _T]:
  """一个装饰器，用于将某些方法排除在转发给 C++ 类的集合之外。"""
  if not isinstance(is_enabled, bool):
    raise TypeError("``is_enabled`` must be a bool")
  def decorator(f):
    if is_enabled:
      original_func = _original_func(f)
      original_func._use_cpp = True  # pyrefly: ignore[missing-attribute]
    return f
  return decorator


class StrictABCMeta(abc.ABCMeta):
  """`abc.ABCMeta` 的一个变体，它不允许虚子类。

  支持虚子类要求 `abc.ABCMeta` 在进行实例/子类检查时
  经由纯 Python 往返。对于需要虚子类的 ABC 而言这没有问题，
  但对于不需要虚子类的 ABC 来说则是一种浪费。
  """
  def register(cls, subclass):
    del subclass  # 未使用。
    raise NotImplementedError(f"{cls} does not support virtual subclasses")

  __instancecheck__ = type.__instancecheck__
  __subclasscheck__ = type.__subclasscheck__


class StrictABC(metaclass=StrictABCMeta):
  __slots__ = ()


test_event_listener: Callable | None = None

def test_event(name: str, *args) -> None:
  if not test_event_listener:
    return
  test_event_listener(name, *args)

Mutex = jaxlib_utils.Mutex


def pprint_bytes(num_bytes: int | float) -> str:
  prefixes = ("", "K", "M", "G", "T")
  if num_bytes <= 0:
    return "0.00B"
  exponent = min(math.floor(math.log(num_bytes, 1000)), len(prefixes) - 1)
  scaled_value = num_bytes / (1000**exponent)
  return f"{scaled_value:.2f}{prefixes[exponent]}B"

install_failure_signal_handler = jaxlib_utils.install_failure_signal_handler
