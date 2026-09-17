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

# TODO(jakevdp): fix import cycles and import Array.
Array = Any


if TYPE_CHECKING:
  # safe_zip cannot yet be fully annotated, so we use a strategy similar
  # to that used for builtins.zip in python/typeshed. This supports
  # return types matching input types for up to three arguments.
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
    Like builtin :func:`zip`, but with additional safety checks.

    The differences from :func:`zip` are:

    - :func:`safe_zip` checks that at least one argument is provided.
    - :func:`safe_zip` checks that all arguments have the same length.
    - :func:`safe_zip` returns an eagerly-evaluated list instead of a
      lazily-evaluated iterator.
    """
    if not args:
      raise TypeError("safe_zip requires at least 1 argument.")
    return list(zip(*args, strict=True))
else:
  safe_zip = jaxlib_utils.safe_zip


if TYPE_CHECKING:
  # safe_map cannot yet be fully annotated, so we use a strategy similar
  # to that used for builtins.map in python/typeshed. This supports
  # checking input types for the callable with up to three arguments.
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
  """Unzip sequence of length-2 tuples into two tuples."""
  # Note: we deliberately don't use zip(*xys) because it is lazily evaluated,
  # is too permissive about inputs, and does not guarantee a length-2 output.
  xs: list[T1] = []
  ys: list[T2] = []
  for x, y in xys:
    xs.append(x)
    ys.append(y)
  return tuple(xs), tuple(ys)

def unzip3[T1, T2, T3](xyzs: Iterable[tuple[T1, T2, T3]]
    ) -> tuple[tuple[T1, ...], tuple[T2, ...], tuple[T3, ...]]:
  """Unzip sequence of length-3 tuples into three tuples."""
  # Note: we deliberately don't use zip(*xyzs) because it is lazily evaluated,
  # is too permissive about inputs, and does not guarantee a length-3 output.
  xs: list[T1] = []
  ys: list[T2] = []
  zs: list[T3] = []
  for x, y, z in xyzs:
    xs.append(x)
    ys.append(y)
    zs.append(z)
  return tuple(xs), tuple(ys), tuple(zs)

def subvals[T](lst: Sequence[T], replace: Iterable[tuple[int, T]]) -> tuple[T, ...]:
  """Substitute values within a list."""
  lst = list(lst)
  for i, v in replace:
    lst[i] = v
  return tuple(lst)

def split_list[T](args: Sequence[T], ns: Sequence[int]) -> list[list[T]]:
  """Split list into sublists of the specified sizes."""
  args = list(args)
  lists = []
  for n in ns:
    lists.append(args[:n])
    args = args[n:]
  lists.append(args)
  return lists

def split_list_checked[T](args: Sequence[T], ns: Sequence[int]) -> list[list[T]]:
  """Split list into sublists of the specified sizes."""
  args = list(args)
  assert sum(ns) == len(args) and all(n >= 0 for n in ns)
  lists = []
  for n in ns:
    lists.append(args[:n])
    args = args[n:]
  return lists

def partition_list[T](bs: Sequence[bool], l: Sequence[T]) -> tuple[list[T], list[T]]:
  """Partition a list into two based on a mask."""
  assert len(bs) == len(l)
  lists: tuple[list[T], list[T]] = ([], [])
  for b, x in zip(bs, l):
    lists[b].append(x)
  return lists

def merge_lists[T1, T2](bs: Sequence[bool], l0: Sequence[T1], l1: Sequence[T2]
                ) -> list[T1 | T2]:
  """Merge the elements of two lists based on a mask."""
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
  """Concatenates/flattens a list of lists."""
  return list(it.chain.from_iterable(xs))

flatten = concatenate

_unflatten_done = object()

def unflatten[T](xs: Iterable[T], ns: Sequence[int]) -> list[list[T]]:
  """Splits `xs` into subsequences of lengths `ns`.

  Unlike `split_list`, the `sum(ns)` must be equal to `len(xs)`."""
  xs_iter = iter(xs)
  unflattened = [[next(xs_iter) for _ in range(n)] for n in ns]
  assert next(xs_iter, _unflatten_done) is _unflatten_done
  return unflattened


def curry(f):
  """Curries arguments of f, returning a function on any remaining arguments.

  For example:
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

# Maps caches to the name of the callable they apply to. All caches in
# this dictionary support `cache_clear()`.
_caches: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()

def register_cache(cache: Any, for_what: str):
  """Registers a cache with JAX's cache management.

  Args:
    cache: an object supporting `cache_clear()`, `cache_info()`, and
      `cache_keys()`, like the result of `functools.lru_cache()`.
    for_what: a string to identify what this cache is used for. This is
       used for debugging.
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
  Least recently used cache decorator with weakref support.

  The cache will take a weakref to the first argument of the wrapped function
  and strong refs to all other arguments. In all other respects it should
  behave similar to `functools.lru_cache`. The cache is thread local.
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


# Interner from strong keys to weak values, intended for us to intern object
# construction, thereby making subsequent __eq__ and __hash__ calls cheap and
# based on object identity.
#
# Caution: The interner does not know about the *signature* of the cached
# function. In particular, if the same argument value can be passed as either
# an arg or a kwarg, then the interner may store multiple entries for the same
# logical call. If this troubles you canonicalize the arguments first, e.g.
# via a wrapper function.
weak_value_interner = lib_weakref_lru_cache.weak_value_interner


def immutable(cls):
  """Decorator to avoid boilerplate for immutable interned classes."""
  def __deepcopy__(self, memo):
    # Deep copy of a singleton interned object is the identity.
    return self
  cls.__deepcopy__ = __deepcopy__

  # Pickling calls __getstate__ and __setstate__, but we're assuming the
  # caller will implement __getnewargs_ex__.
  def __getstate__(self):
    return None
  def __setstate__(self, state):
    pass
  cls.__getstate__ = __getstate__
  cls.__setstate__ = __setstate__

  # Discourage mutation after construction.
  def __setattr__(self, name, value):
    raise AttributeError(f"cannot assign to field {name!r}")
  def __delattr__(self, name):
    raise AttributeError(f"cannot delete field {name!r}")
  cls.__setattr__ = __setattr__
  cls.__delattr__ = __delattr__
  return cls


# The types of arguments for which `multi_weakref_lru_cache` should keep
# weak references.
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
  """Least recently used cache decorator with weakref support.

  Similar to `weakref_lru_cache`, except that it keeps weak references
  to all positional and keyword arguments for which
  `is_weakref_cache_key_type()` is true, and strong references to
  other arguments. The cache entry is removed if any of the weakref
  arguments dies.
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


