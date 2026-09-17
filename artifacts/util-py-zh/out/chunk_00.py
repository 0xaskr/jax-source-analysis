# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import abc
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
import functools
from functools import partial
import itertools as it
import logging
import math
import operator
from typing import (Any, ParamSpec, Protocol, SupportsIndex,
                    TypeVar, overload, TYPE_CHECKING, cast)
import weakref

import numpy as np

from jax._src import config
from jax._src.lib import pytree as lib_pytree
from jax._src.lib import weakref_lru_cache as lib_weakref_lru_cache
from jax._src.lib import utils as jaxlib_utils

logger = logging.getLogger(__name__)

Seq = Sequence

# TODO(jakevdp): 修复导入循环并导入 Array。
Array = Any


if TYPE_CHECKING:
  # safe_zip 目前还无法完整地标注类型，因此我们采用与 python/typeshed
  # 中 builtins.zip 类似的策略。这支持最多三个参数时返回类型
  # 与输入类型相匹配。
  @overload
  def safe_zip[T1](__arg1: Iterable[T1], /) -> list[tuple[T1]]:
    ...
  @overload
  def safe_zip[T1, T2](__arg1: Iterable[T1], __arg2: Iterable[T2], /) -> list[tuple[T1, T2]]:
    ...
  @overload
  def safe_zip[T1, T2, T3](__arg1: Iterable[T1], __arg2: Iterable[T2], __arg3: Iterable[T3], /) -> list[tuple[T1, T2, T3]]:
    ...
  @overload
  def safe_zip(__arg1: Iterable[Any], __arg2: Iterable[Any], __arg3: Iterable[Any], __arg4: Iterable[Any], /, *args) -> list[tuple[Any, ...]]:
    ...

  def safe_zip(*args):
    """
    类似内置的 :func:`zip`，但带有额外的安全检查。

    与 :func:`zip` 的区别在于：

    - :func:`safe_zip` 检查至少提供了一个参数。
    - :func:`safe_zip` 检查所有参数具有相同的长度。
    - :func:`safe_zip` 返回立即求值的列表，而不是
      惰性求值的迭代器。
    """
    if not args:
      raise TypeError("safe_zip requires at least 1 argument.")
    return list(zip(*args, strict=True))
else:
  safe_zip = jaxlib_utils.safe_zip


if TYPE_CHECKING:
  # safe_map 目前还无法完整地标注类型，因此我们采用与 python/typeshed
  # 中 builtins.map 类似的策略。这支持对最多三个参数的
  # 可调用对象进行输入类型检查。
  @overload
  def safe_map[T, T1](f: Callable[[T1], T], __arg1: Iterable[T1], /) -> list[T]: ...

  @overload
  def safe_map[T, T1, T2](f: Callable[[T1, T2], T], __arg1: Iterable[T1], __arg2: Iterable[T2], /) -> list[T]: ...

  @overload
  def safe_map[T, T1, T2, T3](f: Callable[[T1, T2, T3], T], __arg1: Iterable[T1], __arg2: Iterable[T2], __arg3: Iterable[T3], /) -> list[T]: ...

  @overload
  def safe_map[T](f: Callable[..., T], __arg1: Iterable[Any], __arg2: Iterable[Any], __arg3: Iterable[Any], __arg4: Iterable[Any], /, *args) -> list[T]: ...

  def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
      assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(map(f, *args))

else:
  safe_map = jaxlib_utils.safe_map

if TYPE_CHECKING:
  @overload
  def foreach[T1](f: Callable[[T1], Any], __arg1: Iterable[T1], /) -> None: ...

  @overload
  def foreach[T1, T2](f: Callable[[T1, T2], Any], __arg1: Iterable[T1], __arg2: Iterable[T2], /) -> None: ...

  @overload
  def foreach[T1, T2, T3](f: Callable[[T1, T2, T3], Any], __arg1: Iterable[T1], __arg2: Iterable[T2], __arg3: Iterable[T3], /) -> None: ...

  @overload
  def foreach(f: Callable[..., Any], __arg1: Iterable[Any], __arg2: Iterable[Any], __arg3: Iterable[Any], __arg4: Iterable[Any], /, *args) -> None: ...

  def foreach(f, *args):
    safe_map(f, *args)
    return None

else:
  foreach = jaxlib_utils.foreach


def unzip2[T1, T2](xys: Iterable[tuple[T1, T2]]
    ) -> tuple[tuple[T1, ...], tuple[T2, ...]]:
  """将长度为 2 的元组序列解包为两个元组。"""
  # Note: 我们有意不使用 zip(*xys)，因为它是惰性求值的，
  # 对输入过于宽松，并且不保证输出长度为 2。
  xs: list[T1] = []
  ys: list[T2] = []
  for x, y in xys:
    xs.append(x)
    ys.append(y)
  return tuple(xs), tuple(ys)

def unzip3[T1, T2, T3](xyzs: Iterable[tuple[T1, T2, T3]]
    ) -> tuple[tuple[T1, ...], tuple[T2, ...], tuple[T3, ...]]:
  """将长度为 3 的元组序列解包为三个元组。"""
  # Note: 我们有意不使用 zip(*xyzs)，因为它是惰性求值的，
  # 对输入过于宽松，并且不保证输出长度为 3。
  xs: list[T1] = []
  ys: list[T2] = []
  zs: list[T3] = []
  for x, y, z in xyzs:
    xs.append(x)
    ys.append(y)
    zs.append(z)
  return tuple(xs), tuple(ys), tuple(zs)

def subvals[T](lst: Sequence[T], replace: Iterable[tuple[int, T]]) -> tuple[T, ...]:
  """在列表中替换取值。"""
  lst = list(lst)
  for i, v in replace:
    lst[i] = v
  return tuple(lst)

def split_list[T](args: Sequence[T], ns: Sequence[int]) -> list[list[T]]:
  """将列表切分为指定大小的子列表。"""
  args = list(args)
  lists = []
  for n in ns:
    lists.append(args[:n])
    args = args[n:]
  lists.append(args)
  return lists

def split_list_checked[T](args: Sequence[T], ns: Sequence[int]) -> list[list[T]]:
  """将列表切分为指定大小的子列表。"""
  args = list(args)
  assert sum(ns) == len(args) and all(n >= 0 for n in ns)
  lists = []
  for n in ns:
    lists.append(args[:n])
    args = args[n:]
  return lists

def partition_list[T](bs: Sequence[bool], l: Sequence[T]) -> tuple[list[T], list[T]]:
  """根据掩码将列表划分为两部分。"""
  assert len(bs) == len(l)
  lists: tuple[list[T], list[T]] = ([], [])
  for b, x in zip(bs, l):
    lists[b].append(x)
  return lists

def merge_lists[T1, T2](bs: Sequence[bool], l0: Sequence[T1], l1: Sequence[T2]
                ) -> list[T1 | T2]:
  """根据掩码合并两个列表的元素。"""
  assert sum(bs) == len(l1) and len(bs) - sum(bs) == len(l0)
  i0, i1 = iter(l0), iter(l1)
  out: list[T1 | T2] = [next(i1) if b else next(i0) for b in bs]
  assert next(i0, sentinel := object()) is next(i1, sentinel) is sentinel
  return out

def subs_list[T](
    subs: Sequence[int | None], src: Sequence[T], base: Sequence[T],
) -> list[T]:
  base_ = iter(base)
  out = [src[i] if i is not None else next(base_) for i in subs]
  assert next(base_, sentinel := object()) is sentinel
  return out

def subs_list2[T](
    subs1: Sequence[int | None], subs2: Sequence[int | None],
    src1: Sequence[T], src2: Sequence[T], base: Sequence[T],
) -> list[T]:
  assert len(subs1) == len(subs2)
  base_ = iter(base)
  out = [src1[f1] if f1 is not None else src2[f2] if f2 is not None else
         next(base_) for f1, f2, in zip(subs1, subs2)]
  assert next(base_, sentinel := object()) is sentinel
  return out

def concatenate[T](xs: Iterable[Sequence[T]]) -> list[T]:
  """拼接/展平由列表构成的列表。"""
  return list(it.chain.from_iterable(xs))

flatten = concatenate

_unflatten_done = object()

def unflatten[T](xs: Iterable[T], ns: Sequence[int]) -> list[list[T]]:
  """将 `xs` 切分为长度分别为 `ns` 的子序列。

  与 `split_list` 不同，`sum(ns)` 必须等于 `len(xs)`。"""
  xs_iter = iter(xs)
  unflattened = [[next(xs_iter) for _ in range(n)] for n in ns]
  assert next(xs_iter, _unflatten_done) is _unflatten_done
  return unflattened


def curry(f):
  """对 f 的参数进行柯里化，返回一个作用于剩余参数的函数。

  例如：
  >>> f = lambda x, y, z, w: x * y + z * w
  >>> f(2,3,4,5)
  26
  >>> curry(f)(2)(3, 4, 5)
  26
  >>> curry(f)(2, 3)(4, 5)
  26
  >>> curry(f)(2, 3, 4, 5)()
  26
  """
  return wraps(f)(partial(partial, f))

toposort: Callable[[Iterable[Any]], list[Any]]
toposort = partial(jaxlib_utils.topological_sort, "parents")


def cache(max_size=4096, trace_context_in_key: bool | Callable = True, num_shards=64):
  def decorator(f):
    context_fn = (trace_context_in_key if callable(trace_context_in_key)
                  else config.trace_context if trace_context_in_key else None)
    cached_f = lib_weakref_lru_cache.strong_lru_cache(
        f, context_fn, max_size, num_shards=num_shards)
    register_cache(cached_f, str(f))
    return cached_f
  return decorator

# 将缓存映射到其所适用的可调用对象的名称。此字典中的
# 所有缓存都支持 `cache_clear()`。
_caches: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()

def register_cache(cache: Any, for_what: str):
  """向 JAX 的缓存管理注册一个缓存。

  Args:
    cache: 一个支持 `cache_clear()`、`cache_info()` 和
      `cache_keys()` 的对象，例如 `functools.lru_cache()` 的结果。
    for_what: 用于标识此缓存用途的字符串。它
       用于调试。
  """
  _caches[cache] = for_what

def clear_all_caches():
  for cache in list(_caches.keys()):
    cache.cache_clear()

memoize = cache(max_size=None)

def _ignore(): return None

class WeakrefCachedFunc[**P, R](Protocol):
  def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
  def cache_clear(self) -> None: ...
  def cache_info(self) -> lib_weakref_lru_cache.WeakrefLRUCache.WeakrefLRUCacheInfo: ...
  def cache_keys(self) -> list[Any]: ...
  def evict_weakref(self, arg0: Any) -> None: ...

_P = ParamSpec("_P")
_R = TypeVar("_R", covariant=True)

@overload
def weakref_lru_cache[**P, R](
    f: Callable[P, R], /, *, maxsize: int | None = 2048,
    trace_context_in_key: bool = True, explain: Callable | None = None
) -> WeakrefCachedFunc[P, R]: ...

@overload
def weakref_lru_cache(
    f: None = None, /, *, maxsize: int | None = 2048,
    trace_context_in_key: bool = True, explain: Callable | None = None
) -> Callable[[Callable[_P, _R]], WeakrefCachedFunc[_P, _R]]: ...

def weakref_lru_cache[**P, R](
    f: Callable[P, R] | None = None, *, maxsize: int | None = 2048,
    trace_context_in_key: bool = True, explain: Callable | None = None
):
  """
  支持弱引用的最近最少使用（LRU）缓存装饰器。

  缓存会对被包装函数的第一个参数持有弱引用，
  对所有其他参数持有强引用。在其他所有方面，它的行为
  都应类似于 `functools.lru_cache`。该缓存是线程局部的。
  """
  kwargs = dict(maxsize=maxsize, trace_context_in_key=trace_context_in_key,
                explain=explain)
  if f is None:
    return lambda g: _weakref_lru_cache(g, **kwargs)
  return _weakref_lru_cache(f, **kwargs)

def _weakref_lru_cache(f, maxsize, trace_context_in_key, explain):
  cached_f = lib_weakref_lru_cache.weakref_lru_cache(
      config.trace_context if trace_context_in_key else _ignore, f, maxsize,
      explain = lambda: explain if config.explain_cache_misses.value else None)
  register_cache(cached_f, str(f))
  return cached_f


# 从强键到弱值的驻留器（interner），用于对对象构造进行驻留，
# 从而使后续的 __eq__ 和 __hash__ 调用既廉价又
# 基于对象同一性。
#
# 注意：该驻留器并不知道被缓存函数的 *签名*。
# 特别地，如果同一个参数值既可以作为位置参数也可以作为
# 关键字参数传入，那么驻留器可能会为同一次逻辑调用存储多个条目。
# 如果这给你带来困扰，请先规范化这些参数，例如
# 通过一个包装器函数。
weak_value_interner = lib_weakref_lru_cache.weak_value_interner


def immutable(cls):
  """用于避免不可变驻留类样板代码的装饰器。"""
  def __deepcopy__(self, memo):
    # 对单例驻留对象进行深拷贝得到的是它本身。
    return self
  cls.__deepcopy__ = __deepcopy__

  # pickle 会调用 __getstate__ 和 __setstate__，但我们假定
  # 调用方会实现 __getnewargs_ex__。
  def __getstate__(self):
    return None
  def __setstate__(self, state):
    pass
  cls.__getstate__ = __getstate__
  cls.__setstate__ = __setstate__

  # 阻止构造之后发生修改。
  def __setattr__(self, name, value):
    raise AttributeError(f"cannot assign to field {name!r}")
  def __delattr__(self, name):
    raise AttributeError(f"cannot delete field {name!r}")
  cls.__setattr__ = __setattr__
  cls.__delattr__ = __delattr__
  return cls


# `multi_weakref_lru_cache` 应为其保留弱引用的
# 参数类型。
weakref_cache_key_types: set[type] = set()


_multi_weakref_registry = lib_pytree.PyTreeRegistry(
    enable_none=False,
    enable_tuple=True,
    enable_namedtuple=False,
    enable_list=False,
    enable_dict=True,
)


def multi_weakref_lru_cache(
    call: Callable,
    *,
    maxsize: int | None = 2048,
    trace_context_in_key: bool = True,
):
  """支持弱引用的最近最少使用缓存装饰器。

  类似于 `weakref_lru_cache`，区别在于它会对所有
  `is_weakref_cache_key_type()` 为真的位置参数和
  关键字参数保留弱引用，而对其他参数保留强引用。
  如果任何一个弱引用参数消亡，该缓存条目
  就会被移除。
  """
  cached_call = lib_weakref_lru_cache.multi_weakref_lru_cache(
      config.trace_context if trace_context_in_key else _ignore,
      call,
      maxsize=maxsize,
      explain=None,
      registry=_multi_weakref_registry,
      weak_types=weakref_cache_key_types,
  )
  register_cache(cached_call, str(call))
  return cached_call


