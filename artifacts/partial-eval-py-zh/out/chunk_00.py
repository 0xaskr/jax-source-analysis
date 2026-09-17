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

from collections import namedtuple
from collections.abc import Callable, Sequence
import contextlib
from dataclasses import dataclass
from functools import partial
import itertools as it
import logging
import operator as op
from typing import Any, NamedTuple
from weakref import ReferenceType, WeakValueDictionary, finalize, ref

import numpy as np

from jax._src import ad_util
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import effects
from jax._src import linear_util as lu
from jax._src import profiler
from jax._src import source_info_util
from jax._src import tree_util
from jax._src.core import (
    Trace, Tracer, TraceTag, Jaxpr, Literal, AbstractValue,
    new_jaxpr_eqn, Var, DropVar, Atom, JaxprEqn, Primitive,
    get_referent, JaxprEqnContext, typeof)
from jax._src.lib import _jax
from jax._src.source_info_util import SourceInfo
from jax._src.state.types import AbstractRef, ReadEffect
from jax._src import flattree as ft
from jax._src.tree_util import PyTreeDef
from jax._src.util import (unzip2, safe_zip, safe_map, toposort, split_list,
                           merge_lists, partition_list, OrderedSet,
                           weakref_lru_cache, multi_weakref_lru_cache,
                           foreach, test_event)

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip
def identity(x): return x

TracerId = int
AvalId = int
ConstId = int

AttrKind = Any
PyTree = Any
logger = logging.getLogger(__name__)

TracebackScope = _jax.TracebackScope

class PartialVal(tuple):
  """部分值：要么是已知值，要么是未知（抽象）值。

  表示为 `(aval_opt, const)` 二元组，属于以下两种之一：
  * `(None, <Constant>)` 表示已知值，其中该常量满足
    `core.valid_jaxtype(const)`；
  * `(<AbstractValue>, None)` 表示未知值，其特征由
    一个抽象值刻画。
  """
  def __new__(cls, xs: tuple[AbstractValue | None, core.Value]):
    pv, const = xs
    if config.enable_checks.value:
      # 类型检查
      assert isinstance(pv, (AbstractValue, type(None))), xs
      assert (const is None or core.valid_jaxtype(const)), const
      # 不变量检查
      assert (pv is None) ^ (const is None)
    return tuple.__new__(cls, xs)

  @classmethod
  def known(cls, const: core.Value) -> PartialVal:
    return PartialVal((None, const))

  @classmethod
  def unknown(cls, aval: AbstractValue) -> PartialVal:
    return PartialVal((aval, None))

  def is_known(self) -> bool:
    return self[0] is None

  def get_known(self) -> core.Value | None:
    """获取已知值，若已知则返回值，否则返回 None。"""
    return self[1] if self[0] is None else None

  def get_aval(self) -> AbstractValue:
    """直接获取 AbstractValue（若未知），或从常量获取（若已知）。"""
    known = self.get_known()
    if known is not None:
      return typeof(known)
    else:
      return self[0]

@dataclass(frozen=True, slots=True)
class EffectHandle:
  parents : list[Tracer]
  recipe : JaxprEqnRecipe

class JaxprTrace(Trace):

  def __init__(self, parent_trace:Trace, name_stack: source_info_util.NameStack, tag:TraceTag):
    super().__init__()
    self.name_stack = name_stack
    self.tag = tag
    self.parent_trace = parent_trace
    self.requires_low = False
    self.effect_handles : list[EffectHandle] = []
    self.counter = it.count()

  def to_jaxpr_tracer(self, x):
    if isinstance(x, JaxprTracer) and x._trace.tag is self.tag:
      if x._trace is self:
        return x
      else:
        return JaxprTracer(self, x.pval, FreeVar(x))
    else:
      return self.new_const(x)

  def stage_value(self, val):
    return self.to_jaxpr_tracer(val)

  def new_const(self, val) -> JaxprTracer:
    return JaxprTracer(self, PartialVal.known(val), None)

  def new_instantiated_literal(self, val) -> JaxprTracer:
    aval = typeof(val)
    return JaxprTracer(self, PartialVal.unknown(aval), Literal(val, aval))

  def new_instantiated_const(self, val) -> JaxprTracer:
    aval = typeof(val)
    return JaxprTracer(self, PartialVal.unknown(aval), ConstVar(val))

  def new_arg(self, pval: PartialVal) -> JaxprTracer:
    const = pval.get_known()
    # XXX: 在修改这个常量参数剪枝之前请三思！
    # 这对 partial_eval_jaxpr 有着极其重要的影响。
    # 最重要的是，这保证了未知 jaxpr 绝不会使用
    # 已知输入（如果它需要这些输入，它们会作为残差传递过去）。
    if const is None:
      aval = pval.get_aval()
      return JaxprTracer(self, PartialVal.unknown(aval), LambdaBinding())
    else:
      return self.new_const(const)

  def instantiate_const(self, tracer: JaxprTracer) -> JaxprTracer:
    const = tracer.pval.get_known()
    if const is None:
      return tracer
    else:
      if core.is_literalable(const, True):
        return self.new_instantiated_literal(const)
      else:
        return self.new_instantiated_const(const)

  def process_primitive(self, primitive, tracers, params, /):
    with core.set_current_trace(self.parent_trace):
      if primitive in custom_partial_eval_rules:
        tracers = map(self.to_jaxpr_tracer, tracers)
        return custom_partial_eval_rules[primitive](self, *tracers, **params)
      else:
        return self.default_process_primitive(primitive, tracers, params)

  def default_process_primitive(self, primitive, tracers, params):
    # 默认情况下，如果所有输入追踪器都是已知的，就绑定该原语
    # 并认为所有输出都是已知的。否则，把该应用暂存到
    # jaxpr 中，并认为所有输出都是未知的。
    tracers = map(self.to_jaxpr_tracer, tracers)
    consts = [t.pval.get_known() for t in tracers]
    if all(c is not None for c in consts):
      return primitive.bind_with_trace(self.parent_trace, consts,
                                       tuple(t.aval for t in tracers), params)
    tracers = map(self.instantiate_const, tracers)
    avals = [t.aval for t in tracers]
    out_aval, effs = primitive.abstract_eval(*avals, **params)
    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    if primitive.multiple_results:
      out_tracers = [JaxprTracer(self, PartialVal.unknown(aval), None)
                     for aval in out_aval]
      eqn = new_eqn_recipe(self, tracers, out_tracers, primitive, params, effs,
                           source)
      if effects.partial_eval_kept_effects.filter_in(effs):
        self.effect_handles.append(EffectHandle(tracers, eqn))  # pyrefly: ignore[bad-argument-type]  # pyrefly#2385
      for t in out_tracers: t.recipe = eqn
      return out_tracers
    else:
      out_tracer = JaxprTracer(self, PartialVal.unknown(out_aval), None)
      eqn = new_eqn_recipe(self, tracers, [out_tracer], primitive,
                           params, effs, source)
      if effects.partial_eval_kept_effects.filter_in(effs):
        self.effect_handles.append(EffectHandle(tracers, eqn))  # pyrefly: ignore[bad-argument-type]  # pyrefly#2385
      out_tracer.recipe = eqn
      return out_tracer

  def _current_truncated_name_stack(self):
    return source_info_util.current_name_stack()[len(self.name_stack):]

  def process_custom_jvp_call(self, prim, fun, jvp, tracers, /, *, symbolic_zeros):
    tracers = map(self.to_jaxpr_tracer, tracers)
    if all(t.is_known() for t in tracers):
      with core.set_current_trace(self.parent_trace):
        vals = [t.pval[1] for t in tracers]
        return prim.bind(*vals, subfuns=(fun, jvp), symbolic_zeros=symbolic_zeros)
    # 我们假定非平凡的部分求值只是为了构建线性
    # 函数，因此不需要保留自定义 JVP 规则。
    del jvp, symbolic_zeros
    with core.set_current_trace(self):
      return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, prim, f, fwd, bwd, tracers, /, *, out_trees, symbolic_zeros):
    tracers = map(self.to_jaxpr_tracer, tracers)
    if all(t.is_known() for t in tracers):
      vals = [t.pval[1] for t in tracers]
      with core.set_current_trace(self.parent_trace):
        return prim.bind(*vals, subfuns=(f, fwd, bwd), out_trees=out_trees,
                         symbolic_zeros=symbolic_zeros)

    tracers = map(self.instantiate_const, tracers)
    in_knowns = (False,) * len(tracers)
    in_avals = tuple(t.aval for t in tracers)
    f_ = trace_to_subjaxpr_nounits2(f, self.tag, f.debug_info, True)
    f_, aux = partial_eval_wrapper_nounits(f_, in_knowns, in_avals)
    params = dict(subfuns=(f_, fwd, bwd), out_trees=out_trees,
                  symbolic_zeros=symbolic_zeros)
    res = prim.bind_with_trace(self.parent_trace, (), (), params)
    out_knowns, out_avals, jaxpr, env = aux()
    assert not any(out_knowns)
    res_tracers = map(self.instantiate_const, map(self.new_const, res))
    env_tracers = map(self.to_jaxpr_tracer, env)
    out_tracers = [JaxprTracer(self, PartialVal.unknown(a), None)
                   for a in out_avals]
    closed_jaxpr = convert_constvars_jaxpr(jaxpr)

    @partial(lu.wrap_init, debug_info=fwd.debug_info)
    @_memoize
    def fwd_jaxpr_thunk(*zeros):
      fwd_ = _interleave_fun(fwd.with_unknown_names(), zeros)
      fwd_jaxpr, _, consts = trace_to_jaxpr_dynamic(fwd_, in_avals)
      return fwd_jaxpr, consts

    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    params = dict(
        call_jaxpr=closed_jaxpr,
        fwd_jaxpr_thunk=fwd_jaxpr_thunk,
        num_consts=len(res) + len(env),
        bwd=bwd,
        out_trees=out_trees,
        symbolic_zeros=symbolic_zeros
    )
    eqn = new_eqn_recipe(self, (*res_tracers, *env_tracers, *tracers),
                         out_tracers, prim, params,
                         core.positional_effects(closed_jaxpr), source)
    for t in out_tracers: t.recipe = eqn
    return out_tracers

def partition_pvals(
    pvals: list[PartialVal]
  ) -> tuple[list[bool], list[AbstractValue], list[Any]]:
  knowns = [pval.is_known()  for pval in pvals                       ]
  avals  = [pval.get_aval()  for pval in pvals if not pval.is_known()]
  consts = [pval.get_known() for pval in pvals if     pval.is_known()]
  return knowns, avals, consts

@lu.transformation_with_aux2
def partial_eval_wrapper_nounits(
    f: Callable,
    store: lu.Store,
    in_knowns: Sequence[bool],
    in_avals: Sequence[AbstractValue],
    *in_consts: Any):
  in_avals_, in_consts_ = iter(in_avals), iter(in_consts)
  in_pvals = [PartialVal.known(next(in_consts_)) if known else
              PartialVal.unknown(next(in_avals_)) for known in in_knowns]
  sentinel = object()
  assert next(in_avals_, sentinel) is next(in_consts_, sentinel) is sentinel
  jaxpr, (*maybe_fwds, out_pvals, res, env) = f(in_pvals)
  out_knowns, out_avals, out_consts = partition_pvals(out_pvals)
  store.store((*maybe_fwds, out_knowns, out_avals, jaxpr, env))
  return (*out_consts, *res)

custom_partial_eval_rules: dict[Primitive, Callable] = {}


def abstract_eval_fun(fun: Callable, *avals,
                      debug_info: core.DebugInfo, **params):
  _, avals_out = trace_to_jaxpr(
      partial(fun, **params), ft.flatten_args(*avals), debug_info)
  assert all(isinstance(aval, AbstractValue) for aval in avals_out)
  return list(avals_out)


class JaxprTracer(Tracer[JaxprTrace]):
  __slots__ = ['pval', 'recipe']

  _trace: JaxprTrace

  def __init__(self, trace: JaxprTrace, pval: PartialVal,
               recipe: JaxprTracerRecipe | None):
    assert isinstance(pval, PartialVal)
    super().__init__(trace, pval.get_aval())
    self.pval = pval
    self.recipe = recipe

  def __repr__(self):
    return f'Traced<{self.aval}:{self._trace}>'

  @property
  def parents(self) -> Sequence[JaxprTracer]:
    if isinstance(self.recipe, JaxprEqnRecipe):
      # TODO broadcast_in_dim 可能会创建一个新的追踪器……
      return self.recipe.in_tracers
    else:
      return []

  def full_lower(self):
    known = self.pval.get_known()
    if known is not None:
      return core.full_lower(known)
    else:
      return self

  def is_known(self):
    return self.pval.is_known()

  def get_referent(self):
    if self.pval.is_known():
      return get_referent(self.pval.get_known())
    elif isinstance(self.recipe, (FreeVar, ConstVar, Literal)):
      return get_referent(self.recipe.val)
    else:
      return self


@profiler.annotate_function
def trace_to_jaxpr_nounits(
    fun: lu.WrappedFun, pvals: Sequence[PartialVal],
    instantiate: bool | Sequence[bool] = False,
  ) -> tuple[Jaxpr, list[PartialVal], list[core.Value]]:
  current_name_stack = source_info_util.current_name_stack()
  with core.take_current_trace() as parent_trace:
    trace = JaxprTrace(parent_trace, current_name_stack, TraceTag())
    with core.ensure_no_leaks(trace):
      fun = trace_to_subjaxpr_nounits(fun, trace, instantiate, fun.debug_info)
      with core.set_current_trace(trace):
        jaxpr, (out_pvals, consts, env) = fun.call_wrapped(pvals)
        assert not env
      del trace, fun
      return jaxpr, out_pvals, consts

# TODO(mattjj): 多余的包装器……？
@lu.transformation2
def trace_to_subjaxpr_nounits(
    f: Callable,
    trace: JaxprTrace,
    instantiate: Sequence[bool] | bool,
    debug_info: core.DebugInfo,
    in_pvals: Sequence[PartialVal]):
  assert all(isinstance(pv, PartialVal) for pv in in_pvals), in_pvals
  out_tracers, jaxpr, out_consts, env = _trace_to_subjaxpr_nounits(
      f, trace, instantiate, in_pvals, debug_info)
  out_pvals = [t.pval for t in out_tracers]
  del out_tracers
  return jaxpr, (out_pvals, out_consts, env)

@lu.transformation2
def trace_to_subjaxpr_nounits2(
    f: Callable,
    tag: TraceTag,
    debug_info: core.DebugInfo,
    instantiate: bool | Sequence[bool],
    in_pvals: Sequence[PartialVal]):
  assert isinstance(tag, TraceTag)
  assert all(isinstance(pv, PartialVal) for pv in in_pvals), in_pvals
  current_name_stack = source_info_util.current_name_stack()
  with core.take_current_trace() as parent_trace:
    trace = JaxprTrace(parent_trace, current_name_stack, tag)
    out_tracers, jaxpr, out_consts, env = _trace_to_subjaxpr_nounits(
        f, trace, instantiate, in_pvals, debug_info)
    out_pvals = [t.pval for t in out_tracers]
    del out_tracers
  return jaxpr, (out_pvals, out_consts, env)

def _trace_to_subjaxpr_nounits_no_lu_2(f: Callable, trace: JaxprTrace,
                               instantiate: Sequence[bool] | bool,
                               in_pvals: Sequence[PartialVal],
                               debug_info: core.DebugInfo):
  in_knowns  = [pval.is_known()     for pval in in_pvals]
  in_consts  = [pval.get_known()    for pval in in_pvals if     pval.is_known()]
  in_tracers = [trace.new_arg(pval) for pval in in_pvals if not pval.is_known()]
  in_args = merge_lists(in_knowns, in_tracers, in_consts)
  with core.set_current_trace(trace):
    ans = f(*in_args)
  assert isinstance(ans, ft.FlatTree), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  assert all(isinstance(x, Tracer) or core.valid_jaxtype(x) for x in ans), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  if isinstance(instantiate, bool):
    instantiate = [instantiate] * len(ans)
  out_tracers = ans.map(trace.to_jaxpr_tracer)
  out_tracers = out_tracers.map2(
      instantiate, lambda t, inst: trace.instantiate_const(t) if inst else t)
  out_tracers_ = [t for t in out_tracers if not t.is_known()]
  jaxpr, out_consts, env = tracers_to_jaxpr(
      in_tracers, out_tracers_, trace.effect_handles,
      debug_info.with_unknown_names())
  return out_tracers, jaxpr, out_consts, env

def _trace_to_subjaxpr_nounits(f: Callable, trace: JaxprTrace,
                               instantiate: Sequence[bool] | bool,
                               in_pvals: Sequence[PartialVal],
                               debug_info: core.DebugInfo):
  in_knowns  = [pval.is_known()     for pval in in_pvals]
  in_consts  = [pval.get_known()    for pval in in_pvals if     pval.is_known()]
  in_tracers = [trace.new_arg(pval) for pval in in_pvals if not pval.is_known()]
  in_args = merge_lists(in_knowns, in_tracers, in_consts)
  with core.set_current_trace(trace):
    ans = f(*in_args)
  assert isinstance(ans, (list, tuple)), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  assert all(core.valid_jaxtype(x) for x in ans), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  if isinstance(instantiate, bool):
    instantiate = [instantiate] * len(ans)
  out_tracers = map(trace.to_jaxpr_tracer, ans)
  out_tracers = [trace.instantiate_const(t) if inst else t
                 for inst, t in zip(instantiate, out_tracers)]
  out_tracers_ = [t for t in out_tracers if not t.is_known()]
  jaxpr, out_consts, env = tracers_to_jaxpr(
      in_tracers, out_tracers_, trace.effect_handles,
      debug_info.with_unknown_names())
  return out_tracers, jaxpr, out_consts, env

# 下面这个变体实现了一项优化：同时作为输入的残差
# 在辅助数据中指明，而不是作为输出传递出去。
# TODO(mattjj): 更新所有调用方以使用这个版本，并删除另一个版本。
