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

    # Which out_consts (aka residuals) are just forwarded inputs? Check obj id.
    in_consts  = [pval.get_known()    for pval in in_pvals if     pval.is_known()]
    id_map = {id(c): i for i, c in enumerate(in_consts)}
    fwds: list[int | None] = [id_map.get(id(c)) for c in out_consts]
    pruned_consts = [c for c, fwd in zip(out_consts, fwds) if fwd is None]

    del out_tracers
  return jaxpr, (fwds, out_pvals, pruned_consts, env)

# The below variant implements two optimizations:
#  1. residuals that are also primal inputs are indicated in aux data rather
#     than passed as outputs;
#  2. residuals that are also primal outputs are indicated in aux data rather
#     than passed as redundant outputs.
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

  # Which consts (aka residuals) are just forwarded inputs? Check obj id.
  in_consts  = [pval.get_known()    for pval in  in_pvals if    pval.is_known()]
  id_map = {id(c): i for i, c in enumerate(in_consts)}
  input_fwds: list[int | None] = [id_map.get(id(c)) for c in consts]

  # Which consts (aka residuals) are already primal outputs? Check obj id.
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
  """Constructs Jaxpr given tracers for inputs and outputs.

  Params:
    in_tracers: the tracers that were created for the function inputs
    out_tracers: the tracers that were output by the function.
    debug_info: the debug info for the function.

  Returns: a triple of a `Jaxpr`, a list of constant values corresponding to
    the `constvars` in the returned Jaxps, and a list of environment values.
    The vars for the environment values have been prepended to the Jaxpr's
    `invars`.
  """
  gensym = core.gensym()

  t_to_var: dict[TracerId, Var] = {}
  consts: dict[Var, Any] = {}
  env: dict[Var, JaxprTracer] = {}
  constid_to_var: dict[ConstId, Var] = {}  # for deduplication

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
      # TODO broadcast_in_dim can create a new tracer, not present in parents
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
  # del getvar  # needed to avoid cyclic-reference closure, apparently!
  return jaxpr, const_vals, env_vals

@weakref_lru_cache
def move_envvars(jaxpr: Jaxpr, which: tuple[bool, ...]) -> Jaxpr:
  """Move the leading invars selected by `which` after the unselected ones."""
  assert not jaxpr.consts
  keep, env = partition_list(which, jaxpr.invars[:len(which)])
  return jaxpr.replace(invars=[*keep, *env, *jaxpr.invars[len(which):]])

def separate_consts(jaxpr: Jaxpr) -> tuple[Jaxpr, list[Any]]:
  """Detaches the consts and returns them explicitly."""
  return convert_constvars_jaxpr(jaxpr), jaxpr.consts

def convert_constvars_jaxpr(jaxpr: Jaxpr) -> Jaxpr:
  """Detaches the consts, exposing the constant inputs as leading invars."""
  return _detach_consts(jaxpr) if jaxpr.consts else jaxpr

@weakref_lru_cache
def _detach_consts(jaxpr: Jaxpr) -> Jaxpr:
  return jaxpr.with_consts(())


def partial_eval_jaxpr_nounits(
    jaxpr: Jaxpr, unknowns: Sequence[bool],
    instantiate: bool | Sequence[bool],
  ) -> tuple[Jaxpr, Jaxpr, list[bool], list[AbstractValue]]:
  """Unzip a jaxpr in two by data dependence into 'known' and 'unknown' parts.

  That is, given a jaxpr and a sequence of booleans indicating which jaxpr
  inputs (i.e. invars) are considered unknown, produce two jaxprs, a list of
  booleans representing which of the original jaxpr's outputs are unknown (i.e.
  have a data dependence on an unknown input), and a list of abstract values
  representing residuals (part of the first jaxpr's output and the second
  jaxpr's input). The two jaxprs result from partitioning the original jaxpr's
  first-order primitive applications based on whether all the inputs to the
  application are known (in which case the application is represented in the
  'known' jaxpr and its result is considered known) or whether any inputs to the
  application are unknown (in which case the application is represented in the
  'unknown' jaxpr and its result is considered unknown). Higher-order primitives
  are recursively unzipped in two.

  The `instantiate` argument can be used to ensure some outputs are lifted into
  the 'unknown' jaxpr.

  For example, give an input jaxpr:

    { lambda ; a:f32[] b:f32[]. let
        c:f32[] = cos a
        d:f32[] = sin a
        e:f32[] = neg d
        f:f32[] = mul e b
      in (c, f) }

  then applying this function with `unknowns=[False, True]` and
  `instantiate=False` produces as an output triple:

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

  Notice in particular that the first output (jaxpr_known) contains all the
  primitive applications which do not have a data dependence on an unknown
  input. Also notice the input and output types: the input type of the first
  jaxpr produced represents the type of the known inputs of the original jaxpr,
  and the output type of the second jaxpr produced represents the type of the
  unknown outputs of the original jaxpr.

  In the above example, the output of jaxpr_known named `d` is a _residual_
  output, and corresponds to the input named `a` in jaxpr_unknown. In general,
  jaxpr_known will produce extra outputs (at the end of its output list)
  corresponding to intermediate values of the original jaxpr which must be
  passed to jaxpr_unknown (as leading inputs).
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
      # If it's an effectful primitive, we always to run and avoid staging it.
      policy = ensure_enum(saveable(
          eqn.primitive, *[x.aval for x in eqn.invars], **eqn.params))
      if has_effects(eqn.effects) or isinstance(policy, SaveableType):
        foreach(partial(write, False, False), eqn.outvars)
      elif isinstance(policy, Offloadable):
        # TODO(slebedev): This is a legit error which requires a BUILD fix.
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
        # resvars are known and available in the backward jaxpr.
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
        # outvars are known and available in the backward jaxpr.
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

  # TODO(mattjj,necula): debug info should be updated here
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
  # TODO(mattjj,necula): debug info should be updated here
  jaxpr_staged = jaxpr.replace(
      invars=staged_invars, outvars=outs_staged, eqns=staged_eqns,
      effects=staged_effects,
      debug_info=jaxpr.debug_info.with_unknown_names())
  config.enable_checks.value and core.check_jaxpr(jaxpr_staged)

  return (jaxpr_known, jaxpr_staged, out_unknowns, out_inst, len(residuals),
          len(non_input_res_refs))


