@lu.transformation2
def trace_to_subjaxpr_nounits_fwd(
    f: Callable,
    tag: TraceTag,
    debug_info: core.DebugInfo,
    instantiate: bool | Sequence[bool],
    in_pvals: Sequence[PartialVal]):
  assert all(isinstance(pv, PartialVal) for pv in in_pvals), in_pvals
  current_name_stack = source_info_util.current_name_stack()
  with core.take_current_trace() as parent_trace:
    trace = JaxprTrace(parent_trace, current_name_stack, tag)
    with core.set_current_trace(trace):
      out_tracers, jaxpr, out_consts, env = _trace_to_subjaxpr_nounits(
          f, trace, instantiate, in_pvals, debug_info)
    out_pvals = [t.pval for t in out_tracers]

    # 哪些 out_consts（即残差）只是被转发的输入？检查对象 id。
    in_consts  = [pval.get_known()    for pval in in_pvals if     pval.is_known()]
    id_map = {id(c): i for i, c in enumerate(in_consts)}
    fwds: list[int | None] = [id_map.get(id(c)) for c in out_consts]
    pruned_consts = [c for c, fwd in zip(out_consts, fwds) if fwd is None]

    del out_tracers
  return jaxpr, (fwds, out_pvals, pruned_consts, env)

# 下面这个变体实现了两项优化：
#  1. 同时也是原始输入的残差在辅助数据中标出，而不是
#     作为输出传递；
#  2. 同时也是原始输出的残差在辅助数据中标出，而不是
#     作为冗余输出传递。
def trace_to_subjaxpr_nounits_fwd2(
    f: Callable,
    tag: TraceTag,
    debug_info: core.DebugInfo,
    instantiate: bool | Sequence[bool],
    in_pvals: Sequence[PartialVal]):
  assert all(isinstance(pv, PartialVal) for pv in in_pvals), in_pvals
  current_name_stack = source_info_util.current_name_stack()
  with core.take_current_trace() as parent_trace:
    trace = JaxprTrace(parent_trace, current_name_stack, tag)
    out_tracers, jaxpr, consts, env = _trace_to_subjaxpr_nounits_no_lu_2(
        f, trace, instantiate, in_pvals, debug_info)
    out_pvals = out_tracers.map(lambda t: t.pval)

  # 哪些 consts（即残差）只是被转发的输入？检查对象 id。
  in_consts  = [pval.get_known()    for pval in  in_pvals if    pval.is_known()]
  id_map = {id(c): i for i, c in enumerate(in_consts)}
  input_fwds: list[int | None] = [id_map.get(id(c)) for c in consts]

  # 哪些 consts（即残差）已经是原始输出？检查对象 id。
  out_consts = [pval.get_known()    for pval in out_pvals if    pval.is_known()]
  id_map = {id(c): i for i, c in enumerate(out_consts)}
  output_fwds: list[int | None] = [id_map.get(id(c)) for c in consts]

  pruned_consts = [c for c, f1, f2 in zip(consts, input_fwds, output_fwds)
                   if f1 is None and f2 is None]

  del out_tracers
  return jaxpr, (input_fwds, output_fwds, out_pvals, pruned_consts, env)


FreeVar = namedtuple('FreeVar', ['val'])
ConstVar = namedtuple('ConstVar', ['val'])
LambdaBinding = namedtuple('LambdaBinding', [])
class JaxprEqnRecipe(NamedTuple):
  eqn_id: Any
  in_tracers: Sequence[JaxprTracer]
  out_tracer_refs: Sequence[ref[JaxprTracer]]
  out_avals: Sequence[core.AbstractValue]
  primitive: Primitive
  params: dict[str, Any]
  effects: core.Effects
  source_info: source_info_util.SourceInfo
  ctx: JaxprEqnContext

JaxprTracerRecipe = (
    JaxprEqnRecipe | LambdaBinding | FreeVar | ConstVar | Literal
)

def new_eqn_recipe(trace: JaxprTrace,
                   in_tracers: Sequence[JaxprTracer],
                   out_tracers: Sequence[JaxprTracer],
                   primitive: Primitive,
                   params: dict[str, Any],
                   effects: core.Effects,
                   source_info: source_info_util.SourceInfo,
                   ctx: JaxprEqnContext | None = None) -> JaxprEqnRecipe:
  out_avals = [t.aval for t in out_tracers]
  ctx = ctx or core.current_jaxpr_eqn_context()
  return JaxprEqnRecipe(next(trace.counter), tuple(in_tracers), map(ref, out_tracers),
                        out_avals, primitive, params, effects, source_info,
                        ctx)

def tracers_to_jaxpr(
  in_tracers: Sequence[JaxprTracer],
  out_tracers: Sequence[JaxprTracer],
  effect_handles: Sequence[Any],
  debug_info: core.DebugInfo,
  ) -> tuple[Jaxpr, tuple[Any, ...], tuple[Any, ...]]:
  """根据输入和输出的追踪器构造 Jaxpr。

  Params:
    in_tracers: 为函数输入创建的追踪器
    out_tracers: 函数输出的追踪器。
    debug_info: 函数的调试信息。

  Returns: 一个三元组，包含一个 `Jaxpr`、一个对应于返回的 Jaxpr 中
    `constvars` 的常量值列表，以及一个环境值列表。
    环境值的变量已被前置到该
    Jaxpr 的 `invars` 中。
  """
  gensym = core.gensym()

  t_to_var: dict[TracerId, Var] = {}
  consts: dict[Var, Any] = {}
  env: dict[Var, JaxprTracer] = {}
  constid_to_var: dict[ConstId, Var] = {}  # 用于去重

  def get_atom(t: JaxprTracer) -> Atom:
    return t.recipe if type(t.recipe) is Literal else t_to_var[id(t)]

  def newvar(t: JaxprTracer | None) -> Var:
    assert t is not None
    var = gensym(t.aval)
    var_ = t_to_var.setdefault(id(t), var)
    assert var is var_
    return var

  processed_eqn_ids = set()
  eqns: list[core.JaxprEqn] = []
  is_high = False

  reachable = toposort
  tracers = reachable((*in_tracers, *out_tracers, *effect_handles))
  def sort_key(t):
    r = t.recipe
    return r.eqn_id if isinstance(r, JaxprEqnRecipe) else -1
  tracers = sorted(tracers, key=sort_key)

  for t in tracers:
    r = t.recipe
    if isinstance(r, JaxprEqnRecipe):
      # TODO broadcast_in_dim 可能创建一个新追踪器，它不在父节点中
      if r.eqn_id not in processed_eqn_ids:
        in_atoms = map(get_atom, r.in_tracers)
        outvars = [DropVar(a) if rf() is None else newvar(rf())
                   for a, rf in zip(r.out_avals, r.out_tracer_refs)]
        eqns.append(new_jaxpr_eqn(in_atoms, outvars, r.primitive, r.params,
                                  r.effects, r.source_info, r.ctx))
        in_avals = [x.aval for x in in_atoms]
        is_high |= r.primitive.is_high(*in_avals, **r.params)
        processed_eqn_ids.add(r.eqn_id)
    elif isinstance(r, LambdaBinding):
      if not any(t is in_tracer for in_tracer in in_tracers):
        raise core.escaped_tracer_error(t, f"Tracer not in input tracers: {t}")
      newvar(t)
    elif isinstance(r, ConstVar):
      var = constid_to_var.get(id(r.val))
      if var is None:
        var = constid_to_var[id(r.val)] = newvar(t)
        consts[var] = r.val
      t_to_var[id(t)] = var
    elif isinstance(r, FreeVar):
      env[newvar(t)] = r.val
    elif isinstance(r, Literal):
      pass
    elif r is None:
      assert False
    else:
      raise TypeError(r)

  env_vars, env_vals = unzip2(env.items())
  invars = [*env_vars, *map(get_atom, in_tracers)]
  const_vars, const_vals = unzip2(consts.items())
  outvars = map(get_atom, out_tracers)
  jaxpr_effects = make_jaxpr_effects(const_vars, invars, outvars, eqns)
  is_high |= any(x.aval.is_high for x in it.chain(const_vars, invars, outvars))
  jaxpr = Jaxpr(const_vars, invars,  # pyrefly: ignore[bad-argument-type]
                outvars, eqns, jaxpr_effects, debug_info, is_high)
  config.enable_checks.value and core.check_jaxpr(jaxpr)
  # del getvar  # 显然需要这样来避免循环引用的闭包！
  return jaxpr, const_vals, env_vals

@weakref_lru_cache
def move_envvars(jaxpr: Jaxpr, which: tuple[bool, ...]) -> Jaxpr:
  """把由 `which` 选中的开头若干 invars 移到未被选中的 invars 之后。"""
  assert not jaxpr.consts
  keep, env = partition_list(which, jaxpr.invars[:len(which)])
  return jaxpr.replace(invars=[*keep, *env, *jaxpr.invars[len(which):]])

def separate_consts(jaxpr: Jaxpr) -> tuple[Jaxpr, list[Any]]:
  """分离出常量并将它们显式返回。"""
  return convert_constvars_jaxpr(jaxpr), jaxpr.consts

def convert_constvars_jaxpr(jaxpr: Jaxpr) -> Jaxpr:
  """分离出常量，把常量输入暴露为开头的 invars。"""
  return _detach_consts(jaxpr) if jaxpr.consts else jaxpr

@weakref_lru_cache
def _detach_consts(jaxpr: Jaxpr) -> Jaxpr:
  return jaxpr.with_consts(())


def partial_eval_jaxpr_nounits(
    jaxpr: Jaxpr, unknowns: Sequence[bool],
    instantiate: bool | Sequence[bool],
  ) -> tuple[Jaxpr, Jaxpr, list[bool], list[AbstractValue]]:
  """按数据依赖把一个 jaxpr 一分为二，拆成“已知”和“未知”两部分。

  也就是说，给定一个 jaxpr 和一个布尔序列，指明哪些 jaxpr
  输入（即 invars）被视为未知，则产出两个 jaxpr、一个
  布尔列表，表示原 jaxpr 的哪些输出是未知的（即
  对某个未知输入存在数据依赖），以及一个抽象值列表，
  这些抽象值代表残差（是第一个 jaxpr 输出的一部分，也是第二个
  jaxpr 的输入）。这两个 jaxpr 来自对原 jaxpr 的
  一阶原语应用所做的划分：依据某个应用的全部输入
  是否都已知（此时该应用被表示在
  “已知”jaxpr 中，其结果也被视为已知），还是其任一输入
  未知（此时该应用被表示在
  “未知”jaxpr 中，其结果也被视为未知）。高阶原语
  会被递归地一分为二。

  `instantiate` 参数可用于确保某些输出被提升到
  “未知”jaxpr 中。

  例如，给定如下输入 jaxpr：

    { lambda ; a:f32[] b:f32[]. let
        c:f32[] = cos a
        d:f32[] = sin a
        e:f32[] = neg d
        f:f32[] = mul e b
      in (c, f) }

  那么以 `unknowns=[False, True]` 和
  `instantiate=False` 应用该函数，会产出如下三元组：

    # jaxpr_known
    { lambda ; a:f32[]. let
       b:f32[] = cos a
       c:f32[] = sin a
       d:f32[] = neg c
     in (b, d) }

    # jaxpr_unknown
    { lambda ; a:f32[] b:f32[]. let c:f32[] = mul b a in (c,) }

    # out_unknowns
    [False, True]

  特别要注意，第一个输出（jaxpr_known）包含所有
  不与未知输入存在数据依赖的
  原语应用。还要注意输入与输出类型：第一个
  jaxpr 产出的输入类型表示原 jaxpr 中已知输入的类型，
  而第二个 jaxpr 产出的输出类型表示原 jaxpr 中
  未知输出的类型。

  在上例中，jaxpr_known 名为 `d` 的输出是一个_残差_
  输出，它对应 jaxpr_unknown 中名为 `a` 的输入。一般来说，
  jaxpr_known 会产生额外的输出（位于其输出列表末尾），
  这些输出对应原 jaxpr 的中间值，它们必须
  被传给 jaxpr_unknown（作为开头的输入）。
  """
  instantiate = tuple(instantiate) if isinstance(instantiate, list) else instantiate
  return _partial_eval_jaxpr_nounits(jaxpr, tuple(unknowns), instantiate, False)[:-1]

def partial_eval_jaxpr_nounits_fwd(
    jaxpr: Jaxpr, unknowns: Sequence[bool],
    instantiate: bool | Sequence[bool],
    fwd: bool | Sequence[bool] = True,
) -> tuple[Jaxpr, Jaxpr, list[bool], list[AbstractValue], list[int | None]]:
  instantiate = tuple(instantiate) if isinstance(instantiate, list) else instantiate
  fwd = tuple(fwd) if isinstance(fwd, list) else fwd
  return _partial_eval_jaxpr_nounits(jaxpr, tuple(unknowns), instantiate, fwd)

@weakref_lru_cache
def _partial_eval_jaxpr_nounits(
    jaxpr: Jaxpr, in_unknowns: Sequence[bool],
    instantiate: bool | Sequence[bool], fwd: bool | Sequence[bool]):
  f = lu.wrap_init(core.jaxpr_as_fun(jaxpr), debug_info=jaxpr.debug_info)

  cell = []
  def fun(*known_vals_in):
    known_vals_in_ = iter(known_vals_in)
    unknown_avals = (a for a, uk in zip(jaxpr.in_avals, in_unknowns) if uk)
    in_pvals = [PartialVal.unknown(next(unknown_avals)) if uk
                else PartialVal.known(next(known_vals_in_)) for uk in in_unknowns]
    assert next(known_vals_in_, None) is next(unknown_avals, None) is None
    jaxpr_unknown_, (fwds, out_pvals, residuals, ()) = trace_to_subjaxpr_nounits_fwd(
        f, TraceTag(), jaxpr.debug_info, instantiate).call_wrapped(in_pvals)
    jaxpr_unknown = convert_constvars_jaxpr(jaxpr_unknown_)
    out_unknowns = [not pval.is_known() for pval in out_pvals]
    if type(fwd) is bool and not fwd:
      residuals_ = iter(residuals)
      residuals = [next(residuals_) if f is None else known_vals_in[f]
                   for f in fwds]
      assert next(residuals_, None) is None
      fwds = [None] * len(fwds)
    else:
      if type(fwd) is tuple:
        fwd_ = [f for f, uk in zip(fwd, in_unknowns) if not uk]
        residuals_, residuals = iter(residuals), []
        fwds = [residuals.append(next(residuals_)) if f is None else
                residuals.append(known_vals_in[f]) if not fwd_[f] else
                f for f in fwds]
      fwds, residuals = _include_consts_in_fwds(jaxpr.consts, fwds, residuals)
    res_avals = [core.typeof(r) for r in residuals]
    cell.append((out_unknowns, jaxpr_unknown, res_avals, fwds))
    known_vals_out = [pval.get_known() for pval in out_pvals if pval.is_known()]
    return [*known_vals_out, *residuals]

  known_avals = [a for a, uk in zip(jaxpr.in_avals, in_unknowns) if not uk]
  closed_jaxpr_known, _ = trace_to_jaxpr(
      fun, ft.flatten_args(*known_avals), f.debug_info.with_unknown_names())
  (out_unknowns, jaxpr_unknown, res_avals, fwds), = cell

  if config.enable_checks.value:
    core.check_jaxpr(closed_jaxpr_known)
    core.check_jaxpr(jaxpr_unknown)

  closed_jaxpr_unknown = jaxpr_unknown
  return closed_jaxpr_known, closed_jaxpr_unknown, out_unknowns, res_avals, fwds

def _include_consts_in_fwds(consts, fwds, residuals):
  if all(f is None for f in fwds):
    return fwds, residuals
  dummys = [object() for _ in range(max(f for f in fwds if f is not None) + 1)]
  residuals_ = iter(residuals)
  residuals = [next(residuals_) if f is None else dummys[f] for f in fwds]
  assert next(residuals_, None) is None
  idxs = {id(x): i for i, x in enumerate((*consts, *dummys))}
  fwds = [idxs.get(id(r)) for r in residuals]
  residuals = [r for r in residuals if id(r) not in idxs]
  return fwds, residuals


def partial_eval_jaxpr_custom(
    jaxpr: Jaxpr,
    in_unknowns: Sequence[bool],
    in_inst: bool | Sequence[bool],
    ensure_out_unknowns: bool | Sequence[bool],
    ensure_out_inst: bool | Sequence[bool],
    saveable: Callable[..., RematCases_],
  ) -> tuple[Jaxpr, Jaxpr, list[bool], list[bool], int]:
  *outs, num_res_ref = partial_eval_jaxpr_stateful(
      jaxpr, in_unknowns, in_inst, ensure_out_unknowns, ensure_out_inst, saveable)
  if num_res_ref:
    raise ValueError("Cannot use `partial_eval_jaxpr_custom` with stateful jaxprs.")
  return *outs,  # pyrefly: ignore[bad-return]

def partial_eval_jaxpr_stateful(
    jaxpr: Jaxpr,
    in_unknowns: Sequence[bool],
    in_inst: bool | Sequence[bool],
    ensure_out_unknowns: bool | Sequence[bool],
    ensure_out_inst: bool | Sequence[bool],
    saveable: Callable[..., RematCases_] | None,
  ) -> tuple[Jaxpr, Jaxpr, list[bool], list[bool], int, int]:
  if type(in_inst) is bool:
    in_inst = (in_inst,) * len(jaxpr.invars)
  if type(ensure_out_unknowns) is bool:
    ensure_out_unknowns = (ensure_out_unknowns,) * len(jaxpr.outvars)
  if type(ensure_out_inst) is bool:
    ensure_out_inst = (ensure_out_inst,) * len(jaxpr.outvars)
  if saveable is None:
    saveable = everything_saveable
  jaxpr_known, jaxpr_staged, out_unknowns, out_inst, num_res, num_res_ref = \
      _partial_eval_jaxpr_custom_cached(

          jaxpr, tuple(in_unknowns), tuple(in_inst), tuple(ensure_out_unknowns),

          tuple(ensure_out_inst), saveable)
  return jaxpr_known, jaxpr_staged, out_unknowns, out_inst, num_res, num_res_ref

everything_saveable = lambda *_, **__: True

@weakref_lru_cache
def _partial_eval_jaxpr_custom_cached(
    jaxpr: Jaxpr,
    in_unknowns: tuple[bool, ...],
    in_inst: tuple[bool, ...],
    ensure_out_unknowns: tuple[bool, ...],
    ensure_out_inst: tuple[bool, ...],
    saveable: Callable[..., RematCases_],
  ) -> tuple[Jaxpr, Jaxpr, list[bool], list[bool], int, int]:
  env: dict[Var, tuple[bool, bool]] = {}
  residuals: OrderedSet[Var] = OrderedSet()
  residual_refs: OrderedSet[Var] = OrderedSet()

  def read(x: Atom) -> tuple[bool, bool]:
    if type(x) is Var:
      return env[x]
    return (False, True)

  def write(unk: bool, inst: bool, v: Var) -> None:
    assert (unk, inst) != (True, False)
    env[v] = (unk, inst)

  def ensure_instantiated(inst: bool, x: Atom) -> Atom:
    if type(x) is Var and not inst:
      residuals.add(x)
    return x

  def has_effects(effects) -> bool:
    not_really_effects = (core.NamedAxisEffect, core.InternalMutableArrayEffect)
    return any(not isinstance(e, not_really_effects) for e in effects)

  known_eqns, staged_eqns = [], []
  foreach(write, in_unknowns, in_inst, jaxpr.invars)
  foreach(partial(write, False, True), jaxpr.constvars)
  for eqn in jaxpr.eqns:
    unks_in, inst_in = unzip2(map(read, eqn.invars))
    rule = partial_eval_jaxpr_custom_rules.get(eqn.primitive)
    if rule:
      eqn1, eqn2, unks_out, inst_out, res = rule(saveable, unks_in, inst_in, eqn)
      eqn1 and known_eqns.append(eqn1); eqn2 and staged_eqns.append(eqn2)
      for r in res:
        if isinstance(r.aval, AbstractRef):
          residual_refs.add(r)
        else:
          residuals.add(r)
      foreach(write, unks_out, inst_out, eqn.outvars)
    elif any(unks_in):
      inputs = map(ensure_instantiated, inst_in, eqn.invars)
      staged_eqns.append(eqn.replace(invars=inputs))
      foreach(partial(write, True, True), eqn.outvars)
    else:
      known_eqns.append(eqn)
      # 如果它是有副作用的原语，我们总是执行它并避免将它暂存。
      policy = ensure_enum(saveable(
          eqn.primitive, *[x.aval for x in eqn.invars], **eqn.params))
      if has_effects(eqn.effects) or isinstance(policy, SaveableType):
        foreach(partial(write, False, False), eqn.outvars)
      elif isinstance(policy, Offloadable):
        # TODO(slebedev): 这是一个合理错误，需要修改 BUILD 才能解决。
        from jax._src.dispatch import device_put_p, ArrayCopySemantics  # pyrefly: ignore[missing-import]
        resvars = [Var(v.aval.update(memory_space=core.mem_kind_to_space(policy.dst)))
                   for v in eqn.outvars]
        offload_eqn = core.JaxprEqn(
            eqn.outvars, resvars, device_put_p,
            dict(
                devices=(core.mem_kind_to_space(policy.dst),) * len(eqn.outvars),
                srcs=(None,),
                copy_semantics=(ArrayCopySemantics.ALWAYS_COPY,),
            ),
            set(), source_info_util.new_source_info(), core.current_jaxpr_eqn_context())
        known_eqns.append(offload_eqn)
        # resvars 是已知的，并且在反向 jaxpr 中可用。
        foreach(partial(write, False, True), resvars)
        assert all(o.aval.memory_space == core.mem_kind_to_space(policy.src)  # pyrefly: ignore[missing-attribute]
                   for o in eqn.outvars)
        residuals.update(resvars)
        reload_eqn = core.JaxprEqn(
            resvars, eqn.outvars, device_put_p,
            dict(
              devices=(core.mem_kind_to_space(policy.src),) * len(resvars),
              srcs=(None,),
              copy_semantics=(ArrayCopySemantics.ALWAYS_COPY,)
            ),
            set(), source_info_util.new_source_info(), core.current_jaxpr_eqn_context())
        staged_eqns.append(reload_eqn)
        # outvars 是已知的，并且在反向 jaxpr 中可用。
        foreach(partial(write, False, True), eqn.outvars)
      else:
        assert isinstance(policy, RecomputeType)
        inputs = map(ensure_instantiated, inst_in, eqn.invars)
        staged_eqns.append(eqn.replace(invars=inputs))
        foreach(partial(write, False, True), eqn.outvars)
  unzipped = unzip2(map(read, jaxpr.outvars))
  out_unknowns, out_inst = list(unzipped[0]), list(unzipped[1])
  assert all(type(v) is Var for v in residuals), residuals

  for x, inst, ensure_inst in zip(jaxpr.outvars, out_inst, ensure_out_inst):
    if ensure_inst: ensure_instantiated(inst, x)
  out_unknowns = map(op.or_, out_unknowns, ensure_out_unknowns)
  out_inst     = map(op.or_, out_inst,     ensure_out_inst)

  ins_known, _ = partition_list(in_unknowns, jaxpr.invars)
  outs_known, _ = partition_list(out_unknowns, jaxpr.outvars)
  ref_res_is_input = [r in ins_known for r in residual_refs]
  non_input_res_refs, _ = partition_list(ref_res_is_input, list(residual_refs))
  ins_known_and_ref_res = [*ins_known, *non_input_res_refs]
  known_outvars = [*outs_known, *residuals]
  known_effects = make_jaxpr_effects(jaxpr.constvars, ins_known_and_ref_res,
                                     known_outvars, known_eqns)

  # TODO(mattjj,necula): 这里应更新调试信息
  jaxpr_known = jaxpr.replace(
      invars=ins_known_and_ref_res, outvars=known_outvars,
      eqns=known_eqns, effects=known_effects,
      debug_info=jaxpr.debug_info.with_unknown_names())
  config.enable_checks.value and core.check_jaxpr(jaxpr_known)

  _, ins_staged = partition_list(in_inst, jaxpr.invars)
  _, outs_staged = partition_list(out_inst, jaxpr.outvars)
  staged_invars = [*residuals, *non_input_res_refs, *ins_staged]
  staged_effects = make_jaxpr_effects(jaxpr.constvars, staged_invars,
                                      outs_staged, staged_eqns)
  # TODO(mattjj,necula): 这里应更新调试信息
  jaxpr_staged = jaxpr.replace(
      invars=staged_invars, outvars=outs_staged, eqns=staged_eqns,
      effects=staged_effects,
      debug_info=jaxpr.debug_info.with_unknown_names())
  config.enable_checks.value and core.check_jaxpr(jaxpr_staged)

  return (jaxpr_known, jaxpr_staged, out_unknowns, out_inst, len(residuals),
          len(non_input_res_refs))


