def _memoize(fn):
  cells = {}
  sentinel = object()
  def memoized(*args):
    out = cells.get(args, sentinel)
    if out is sentinel:
      with core.set_current_trace(None):
        out = cells[args] = fn(*args)
    return out
  return memoized

@lu.transformation_with_aux2
def _jvp_jaxpr_zeros(f, store, in_zeros, zero_avals, *primal_tangent_avals):
  in_primals, nz_in_tangents = split_list(primal_tangent_avals, [len(in_zeros)])
  symbolic_zeros = map(ad_util.SymbolicZero, zero_avals)
  tangents = merge_lists(in_zeros, nz_in_tangents, symbolic_zeros)
  out = f(*in_primals, *tangents)
  n, ragged = divmod(len(out), 2)
  assert not ragged
  out_primals, out_tangents = out[:n], out[n:]
  out_zeros = [type(t) is ad_util.SymbolicZero for t in out_tangents]
  out_nz_tangents, _ = partition_list(out_zeros, out_tangents)
  store.store(out_zeros)
  return [*out_primals, *out_nz_tangents]

callsites_with_tracing_cache_miss: set[str] = set()

def explain(keys, fun, in_avals, debug_info, *context, **_):
  func_filename = debug_info.func_filename
  if func_filename and not source_info_util.is_user_filename(func_filename):
   return

  msg: list[str] = []
  p = msg.append

  callsite = source_info_util.summarize(source_info_util.current())
  p(f"TRACING CACHE MISS at {callsite}:")

  src_info = ""
  if func_filename:
    src_info += f" defined at {func_filename}"
  if func_lineno := debug_info.func_lineno:
    src_info += f":{func_lineno}"
  func_name = debug_info.func_name

  # have we seen this function before at all?
  keys = [key for fun_ref, *key in keys if fun_ref() is fun]
  if not keys:
    p(f"  never seen function:\n    {func_name} id={id(fun)}{src_info}")
    if callsite in callsites_with_tracing_cache_miss:
      p("  but seen another function defined on the same line; maybe the function is\n"
        "  being re-defined repeatedly, preventing caching?")
    else:
      callsites_with_tracing_cache_miss.add(callsite)
    return logger.log(logging.WARNING, "\n".join(msg))

  p(f"  for {func_name}{src_info}")

  key = (config.trace_context(), (in_avals, debug_info, *context), {})
  min_diff = min(diff_tracing_cache_keys(key, k) for k in keys)[-1]
  p('  all previously seen cache keys differ. For the closest previous key:')
  p('  ' + min_diff)
  return logger.log(logging.WARNING, "\n".join(msg))

def diff_tracing_cache_keys(new_key, old_key) -> tuple[int, int, str]:
  """Explain the diff between two tracing cache keys.
  Returns:
    A tuple of (severity, num_diffs, explanation) for the diff between the two
    keys. Severity is an int, where lower is better.
  """
  new_ctx, (new_tree, new_dbg, *_), () = new_key
  old_ctx, (old_tree, old_dbg, *_), () = old_key
  return (diff_tracing_ctx(new_ctx, old_ctx) or
          diff_trees(new_tree.tree, old_tree.tree) or
          diff_debug(new_dbg, old_dbg) or
          diff_types(new_dbg, new_tree.vals, old_tree.vals) or
          (4, 0, 'cache miss explanation unavailable'))

def diff_tracing_ctx(new_ctx, old_ctx) -> tuple[int, int, str] | None:
  if new_ctx == old_ctx: return None
  diffs: list[str] = []
  msg = "Tracing context doesn't match, e.g. due to config or context manager."
  if len(new_ctx) != len(old_ctx):
    num_diff = abs(len(new_ctx) - len(old_ctx))
    diffs.append("  * number of tracing context values differs: "
                f"now {len(new_ctx)} and before {len(old_ctx)}")
    return 0, num_diff, msg + "\n" + "\n".join(diffs)

  num_diff = sum(map(op.ne, new_ctx, old_ctx))

  context_names = config.trace_context_names()
  if len(context_names) != len(new_ctx):
    diffs.append("  * number of tracing context names differs: "
                  f"context_names {len(context_names)} vs "
                  f"context length {len(new_ctx)}")
    return 0, num_diff, msg + "\n" + "\n".join(diffs)
  for name, new, old in zip(context_names, new_ctx, old_ctx):
    if new != old:
      diffs.append(f"  * {name} differs: now {new} and before {old}")
  return 0, num_diff, msg + "\n" + "\n".join(diffs)

def diff_trees(new_tree, old_tree) -> tuple[int, int, str] | None:
  errs = tree_util.equality_errors_pytreedef(new_tree, old_tree)
  tree_diffs = []
  for path, thing1, thing2, explanation in errs:
    tree_diffs.append(
        f"  * at input path {tree_util.keystr(tuple(path))}, now {thing1} and "
        f"before {thing2}, so {explanation}")
  msg = 'different input pytree:\n' + '\n'.join(tree_diffs)
  if tree_diffs: return 1, len(tree_diffs), msg

def diff_debug(new_dbg, old_dbg) -> tuple[int, int, str] | None:
  msg = "Debug info doesn't match."
  num_diff = sum(map(op.ne, new_dbg, old_dbg))
  if num_diff: return 2, num_diff, msg

def diff_types(dbg, new_leaves, old_leaves) -> tuple[int, int, str] | None:
  if new_leaves == old_leaves: return
  diffs = []
  add_weak_type_hint = False
  for name, new_ty, old_ty in zip(dbg.arg_names, new_leaves, old_leaves):
    if new_ty != old_ty:
      new_str, old_str = new_ty.str_short(True), old_ty.str_short(True)
      if type(new_ty) is type(old_ty) is core.ShapedArray:
        if new_ty.sharding != old_ty.sharding:
          new_str, old_str = new_ty.str_short(True, True), old_ty.str_short(True, True)
        if new_ty.weak_type != old_ty.weak_type:
          add_weak_type_hint = True
          new_str += f'{{weak_type={new_ty.weak_type}}}'
          old_str += f'{{weak_type={old_ty.weak_type}}}'
      diffs.append(f"  * at {name}, now {new_str} and before {old_str}")
  msg = 'different input types:\n' + '\n'.join(diffs)
  if add_weak_type_hint:
    msg += 'https://docs.jax.dev/en/latest/type_promotion.html#weak-types'
  if diffs: return 3, len(diffs), msg


def _lower_debug_info(hi_jaxpr):
  debug_info = hi_jaxpr.debug_info
  if debug_info.arg_names is not None:
    lo_arg_names = tuple(
        name for aval, name in zip(hi_jaxpr.in_avals, debug_info.arg_names)
        for _ in aval.lo_ty())
    debug_info = debug_info._replace(arg_names=lo_arg_names)
  if debug_info.result_paths is not None:
    lo_result_paths = tuple(
    path for aval, path in zip(hi_jaxpr.out_avals, debug_info.result_paths)
        for _ in aval.lo_ty())
    debug_info = debug_info._replace(result_paths=lo_result_paths)
  return debug_info

def trace_to_jaxpr_nocache(
    fun: Callable,
    in_avals: ft.FlatTree,  # (args, kwargs) pair
    debug_info: core.DebugInfo,
    # TODO: let's just make a `trace_to_jaxpr_ft` function for this
    fun_takes_flat_tree_arg=False,
    fun_returns_flat_tree=False,
    requires_low=False,
) -> tuple[Jaxpr, ft.FlatTree]:
  if config.no_tracing.value:
    raise RuntimeError(f"re-tracing function {fun} for "
                       "`jit`, but 'no_tracing' is set")
  test_event("trace_to_jaxpr")
  config.enable_checks.value and debug_info.assert_arg_names(len(in_avals))
  parent_trace = core.trace_ctx.trace
  trace = DynamicJaxprTrace(debug_info, parent_trace=parent_trace,
                            lower=requires_low)
  # Name stack and the traceback scope are reset because the metadata on jaxpr
  # equations should be rooted at the enclosing jaxpr and not contain any
  # context from the callsite. Otherwise metadata from one caller would bleed
  # into metadata from a different caller if we, e.g., inline.
  with (core.ensure_no_leaks(trace), source_info_util.reset_name_stack(),
        TracebackScope()):
    source_info = source_info_util.current()
    if requires_low:
      if debug_info.arg_names is not None:
        debug_info = debug_info._replace(arg_names=tuple(
            name for aval, name in zip(in_avals, debug_info.arg_names)
            for _ in aval.lo_ty()))
      def new_arg(aval):
        lo_tracers = [trace.new_arg(lo_aval, source_info=source_info) for lo_aval in aval.lo_ty()]  # noqa: F821
        return aval.raise_val(*lo_tracers)
      in_tracers = in_avals.map(new_arg)
    else:
      in_tracers = in_avals.map(partial(trace.new_arg, source_info=source_info))

    with core.set_current_trace(trace):
      if fun_takes_flat_tree_arg:
        args_ft, kwargs_ft = in_tracers.unpack()
        assert kwargs_ft.unflatten() == {}  # TODO: handle kwargs
        kwargs = {}
        args = args_ft.unpack()
        del args_ft
      else:
        args, kwargs = in_tracers.unflatten()
      ans_pytree = fun(*args, **kwargs)
      if fun_returns_flat_tree:
        # TODO(dougalm): make result paths optional
        ans = ans_pytree
        debug_info = debug_info.set_result_paths([''] * len(ans))
      else:
        debug_info = debug_info.set_result_paths(ans_pytree)
        ans = ft.flatten(ans_pytree)
      del ans_pytree, args, kwargs

    _check_returned_jaxtypes(debug_info, list(ans))
    ans = ans.map(dtypes.canonicalize_value)
    out_avals = ans.map(typeof)
    if requires_low:
      flat_out_tracers = [trace.to_jaxpr_tracer(x, source_info=source_info)
                          for aval, hi_val in zip(out_avals, ans)
                          for x in aval.lower_val(hi_val)]
      debug_info = debug_info.resolve_result_paths()
      if debug_info.result_paths is not None:
        debug_info = debug_info._replace(result_paths=tuple(
            path for aval, path in zip(out_avals, debug_info.result_paths)
            for _ in aval.lo_ty()))
    else:
      flat_out_tracers = [trace.to_jaxpr_tracer(x, source_info=source_info)
                          for x in ans]

    _check_no_returned_refs(debug_info, list(flat_out_tracers))
    jaxpr, consts = trace.frame.to_jaxpr(trace, list(flat_out_tracers), debug_info,
                                         source_info)
    del trace, fun, in_tracers, flat_out_tracers, ans
  config.enable_checks.value and core.check_jaxpr(jaxpr)
  return jaxpr.with_consts(consts), out_avals


trace_to_jaxpr = weakref_lru_cache(maxsize=None, explain=explain)(
    trace_to_jaxpr_nocache)


# TODO(dougalm): remove in favor of `trace_to_jaxpr`
@profiler.annotate_function
def trace_to_jaxpr_dynamic(
    fun: lu.WrappedFun, in_avals: Sequence[AbstractValue],
    *, keep_inputs: list[bool] | None = None, lower: bool = False,
    auto_dce: bool = False) -> tuple[Jaxpr, list[AbstractValue], list[Any]]:
  config.enable_checks.value and fun.debug_info.assert_arg_names(len(in_avals))
  keep_inputs = [True] * len(in_avals) if keep_inputs is None else keep_inputs
  parent_trace = core.trace_ctx.trace
  trace = DynamicJaxprTrace(fun.debug_info, parent_trace=parent_trace,
                            lower=lower, auto_dce=auto_dce)
  # Name stack and the traceback scope are reset because the metadata on jaxpr
  # equations should be rooted at the enclosing jaxpr and not contain any
  # context from the callsite. Otherwise metadata from one caller would bleed
  # into metadata from a different caller if we, e.g., inline.
  with (core.ensure_no_leaks(trace), source_info_util.reset_name_stack(),
        TracebackScope()):
    source_info = source_info_util.current()
    in_tracers = map(partial(trace.new_arg, source_info=source_info), in_avals)
    in_tracers = [t for t, keep in zip(in_tracers, keep_inputs) if keep]
    with core.set_current_trace(trace):
      ans = fun.call_wrapped(*in_tracers)
    _check_returned_jaxtypes(fun.debug_info, ans)
    ans = map(dtypes.canonicalize_value, ans)
    out_tracers = map(partial(trace.to_jaxpr_tracer, source_info=source_info), ans)
    _check_no_returned_refs(fun.debug_info, out_tracers)
    jaxpr, consts = trace.frame.to_jaxpr(trace, out_tracers, fun.debug_info,
                                         source_info)
    del trace, fun, in_tracers, out_tracers, ans
  config.enable_checks.value and core.check_jaxpr(jaxpr)
  return jaxpr, [v.aval for v in jaxpr.outvars], consts

def _check_returned_jaxtypes(dbg, out_tracers):
  for i, x in enumerate(out_tracers):
    try: typeof(x)
    except TypeError:
      if (dbg and len(paths := dbg.resolve_result_paths().result_paths) > i and
          (p := paths[i].removeprefix('result'))):
        extra = f' at output component {p}'
      else:
        extra = ''
      raise TypeError(
      f"function {dbg.func_src_info} traced for {dbg.traced_for} returned a "
      f"value of type {type(x)}{extra}, which is not a valid JAX type") from None

def _check_no_returned_refs(
    dbg: core.DebugInfo,
    out_tracers: Sequence[DynamicJaxprTracer]
) -> None:
  if not config.mutable_array_checks.value: return
  for i, t in enumerate(out_tracers):
    a = t.aval
    if isinstance(a, AbstractRef):
      result_paths = dbg.resolve_result_paths().safe_result_paths(len(out_tracers))
      if list(result_paths) == ["result"]: result_paths = [""]  # TODO(mattjj): fix in callee
      loc = result_paths[i] and f' at output tree path {result_paths[i]}'
      frame = t._trace.frame
      v = t.val
      eqns = frame.get_eqns()
      # TODO(dougalm): something more efficient
      eqn = next((e for e in eqns if v in e.outvars), None)
      if eqn:
        assert eqn.primitive in (core.ref_p, core.empty_ref_p)
        origin_info = ('\n\nThe returned mutable array was created on line '
                       f'{source_info_util.summarize(eqn.source_info)}.')
      elif v in frame.invars:
        assert isinstance(v, Var)
        arg_name = dbg.safe_arg_names(len(frame.invars))[frame.invars.index(v)]
        origin_info = ('\n\nThe returned mutable array was passed in as the '
                       f'argument {arg_name}.')
      else:
        origin_info = ''
      raise ValueError(
          f"function {dbg.func_src_info} traced for {dbg.traced_for} returned "
          f"a mutable array reference of type {a.str_short()}{loc}, but "
          f"mutable array references cannot be returned.{origin_info}")

class TracerAsName:
  ref: Any
  def __init__(self, tracer):
    self.ref = core.get_referent(tracer)
  def __eq__(self, other):
    return isinstance(other, TracerAsName) and self.ref is other.ref
  def __hash__(self):
    return id(self.ref)

Const = Any
Val = Any


def inline_jaxpr_into_trace(
    trace: DynamicJaxprTrace, src: SourceInfo, jaxpr: Jaxpr,
    consts: Sequence[Any], *arg_tracers: DynamicJaxprTracer) -> list[Any]:
  # This function is conceptually the same thing as just calling eval_jaxpr,
  const_tracers = map(partial(trace.new_const, source_info=src), consts)
  env: dict[Var, DynamicJaxprTracer] = dict(
      zip([*jaxpr.constvars, *jaxpr.invars],
          [*const_tracers, *arg_tracers]))

  def inline_atom(src_, x):
    if isinstance(x, Literal):
      return DynamicJaxprTracer(trace, x.aval, x, src_)
    else:
      return env[x]

  for eqn in jaxpr.eqns:
    src_ = (src if not eqn.source_info.name_stack else
            src.replace(name_stack=src.name_stack + eqn.source_info.name_stack))
    in_tracers = map(partial(inline_atom, src_), eqn.invars)
    out_avals = [v.aval for v in eqn.outvars]

    maybe_consts = try_constant_folding(eqn.primitive, in_tracers, eqn.params, out_avals)
    if maybe_consts is not None:
      out_tracers = [trace.new_const(c, source_info=src_, aval=aval)
                     for c, aval in zip(maybe_consts, out_avals)]
    else:
      effs = {e.replace(env[e.input].val)
              if isinstance(e, effects.JaxprInputEffect) else e
              for e in eqn.effects} if eqn.effects else eqn.effects
      out_tracers = trace.emit_eqn(in_tracers, out_avals, eqn.primitive,
                                   eqn.params, effs, src_, eqn.ctx)
    foreach(env.setdefault, eqn.outvars, out_tracers)

  return map(partial(inline_atom, src), jaxpr.outvars)


def try_constant_folding(primitive, tracers, params, out_avals):
  if primitive in const_fold_rules:
    consts_in = [t.get_const() for t in tracers]
    if any(c is not None for c in consts_in):
      return const_fold_rules[primitive](consts_in, params, out_avals)
  return None

@weakref_lru_cache
def lower_jaxpr2(hi_jaxpr) -> Jaxpr:
  in_avals = ft.flatten(([a.lo_ty() for a in hi_jaxpr.in_avals], {}))
  lo_jaxpr, _ = lower_jaxpr(hi_jaxpr, in_avals)
  return lo_jaxpr

@weakref_lru_cache
def lower_jaxpr(hi_jaxpr: Jaxpr, lo_avals) -> tuple[Jaxpr, ft.FlatTree]:
  env: dict[Var, DynamicJaxprTracer | HTLV] = {}  # noqa # type:ignore

  parent_trace = core.trace_ctx.trace
  trace = DynamicJaxprTrace(hi_jaxpr.debug_info.with_unknown_names(),
                            parent_trace=parent_trace, lower=True)

  def read(src, x):
    if isinstance(x, Literal):
      return x.val
    elif (htlv := env.get(x)) is not None:  # noqa
      return htlv
    else:
      assert not x.aval.is_high
      return trace.var_to_tracer(x, src)  # noqa

  def write(v, x):
    if v.aval.is_high:
      env[v] = x  # noqa: F821
    elif isinstance(x, DynamicJaxprTracer) and x._trace is trace:  # noqa: F821
      env[v] = x  # noqa: F821
    else:
      env[v] = lift_lo_const(v, x)  # noqa: F821

  def lift_lo_const(v, c) -> DynamicJaxprTracer:
    tracer = DynamicJaxprTracer(trace, v.aval, v, src)  # noqa: F821
    trace.frame.constid_to_tracer[id(c)] = tracer  # noqa: F821
    trace.frame.constvar_to_val[v] = c  # noqa: F821
    return tracer

  with (core.ensure_no_leaks(trace), source_info_util.reset_name_stack(),
        TracebackScope()):
    src = source_info_util.current()
    invals = outs = maybe_invals = new_invars = xs = t = None

    lo_avals_lol, () = lo_avals.unflatten()
    for v, xs in zip(hi_jaxpr.invars, lo_avals_lol):
      if v.aval.is_high:
        xs = [trace.new_arg(x, source_info=src) for x in xs]
        env[v] = v.aval.raise_val(*xs)  # type: ignore
      else:
        trace.frame.invars.append(v)

    for v, c in zip(hi_jaxpr.constvars, hi_jaxpr.consts):
      if v.aval.is_high:
        env[v] = c  # treated as an HTLV
      else:
        env[v] = lift_lo_const(v, c)

    with core.set_current_trace(trace):
      eqns = trace.frame.tracing_eqns
      for eqn in hi_jaxpr.eqns:
        maybe_invals = [env.get(x) if isinstance(x, Var) else None for x in eqn.invars]
        hi = eqn.primitive.is_high(*[v.aval for v in eqn.invars], **eqn.params)
        if all(x is None for x in maybe_invals) and not hi:
          eqns.append(eqn)  # type: ignore
        elif not hi:
          new_invars = [x if isinstance(x, Literal) else
                        t.val if (t := env.get(x)) is not None else x
                        for x in eqn.invars]
          eqns.append(eqn.replace(invars=new_invars))
        else:
          invals = map(partial(read, eqn.source_info), eqn.invars)
          name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
          with (source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack),
                eqn.ctx.manager):
            outs = eqn.primitive.to_lojax(*invals, **eqn.params)
          foreach(write, eqn.outvars, outs if eqn.primitive.multiple_results else [outs])

    tracer = partial(trace.to_jaxpr_tracer, source_info=src)
    out_tracers = [dtypes.canonicalize_value(read(src, x)) for x in hi_jaxpr.outvars]
    out_tracers = [v.aval.lower_val2(hi_val).map(tracer)
                  for v, hi_val in zip(hi_jaxpr.outvars, out_tracers)]
    out_tracers = ft.pack(tuple(out_tracers))
    out_avals = out_tracers.map(typeof)
    dbg = _lower_debug_info(hi_jaxpr)
    jaxpr, consts = trace.frame.to_jaxpr(trace, list(out_tracers), dbg, src)
    del trace, env, out_tracers, tracer, read, outs, invals, eqns
    del maybe_invals, new_invars, xs, t

  config.enable_checks.value and core.check_jaxpr(jaxpr)
  assert not any(v.aval.is_high for v in it.chain(jaxpr.constvars, jaxpr.invars))
  return jaxpr.with_consts(consts), out_avals

@weakref_lru_cache
def lower_jaxpr_reference(hi_jaxpr: Jaxpr, lo_avals) -> tuple[Jaxpr, ft.FlatTree]:
  """Reference implementation of lower_jaxpr, for testing and debugging.

  Swap it in with `pe.lower_jaxpr = pe.lower_jaxpr_reference`; all call sites
  look the name up at call time.
  """
  dbg = _lower_debug_info(hi_jaxpr)
  return trace_to_jaxpr(partial(_lower_traceable, hi_jaxpr), lo_avals, dbg,
                        requires_low=True, fun_returns_flat_tree=True)

def _lower_traceable(jaxpr, *lo_args):
  hi_args = [a.raise_val(*xs) for a, xs in zip(jaxpr.in_avals, lo_args)]
  hi_outs = core.jaxpr_as_fun(jaxpr)(*hi_args)
  lo_outs = [a.lower_val2(y) for a, y in zip(jaxpr.out_avals, hi_outs)]
  return ft.pack(tuple(lo_outs))

# vestigial hijax helpers
def raise_lo_outs(hi_avals, lo_outs):
  lo_outs_ = iter(lo_outs)
  hi_outs = [t.raise_val(*it.islice(lo_outs_, len(t.lo_ty()))) for t in hi_avals]
  assert next(lo_outs_, None) is None
  return hi_outs


eval_jaxpr_p = core.eval_jaxpr_p

dce_rules[eval_jaxpr_p] = dce_jaxpr_closed_call_rule
dce_jaxpr_call_rule = dce_jaxpr_closed_call_rule  # alias for downstream users

def _eval_jaxpr_partial_eval(prim, trace, *in_tracers, call_jaxpr, **params):
  in_pvals = [t.pval for t in in_tracers]
  unknown_ins = [not pv.is_known() for pv in in_pvals]
  known_jaxpr, unknown_jaxpr, unknown_outs, res_avals, in_fwd_res = \
      partial_eval_jaxpr_nounits_fwd(call_jaxpr, unknown_ins, instantiate=False)
  consts = [pv.get_known() for pv in in_pvals if pv.is_known()]
  all_known_outs = prim.bind(*consts, call_jaxpr=known_jaxpr, **params)
  known_outs, res = split_list(all_known_outs, [len(all_known_outs) - len(res_avals)])
  res_ = iter(res)
  res = [next(res_) if f is None else [*call_jaxpr.consts, *consts][f]
         for f in in_fwd_res]
  assert next(res_, sentinel := object()) is sentinel
  res_tracers = map(trace.new_instantiated_const, res)
  unk_tracers_in = [t for t in in_tracers if not t.pval.is_known()]
  unk_tracers_out = [JaxprTracer(trace, PartialVal.unknown(aval), None)
                     for aval in unknown_jaxpr.out_avals]
  eqn = new_eqn_recipe(trace, [*res_tracers, *unk_tracers_in], unk_tracers_out,
                       prim, dict(params, call_jaxpr=unknown_jaxpr),
                       core.positional_effects(unknown_jaxpr),
                       source_info_util.current())
  for t in unk_tracers_out: t.recipe = eqn
  if effects.partial_eval_kept_effects.filter_in(unknown_jaxpr.effects):
    trace.effect_handles.append(EffectHandle([*unk_tracers_in, *res_tracers], eqn))  # type: ignore
  return merge_lists(unknown_outs, known_outs, unk_tracers_out)
custom_partial_eval_rules[eval_jaxpr_p] = partial(_eval_jaxpr_partial_eval, eval_jaxpr_p)

partial_eval_jaxpr_custom_rules[eval_jaxpr_p] = \
    partial(closed_call_partial_eval_custom_rule, 'call_jaxpr',
            lambda _, __, ___, ____, _____, ______, x, y: (x, y))

def _lower_and_eval(prim, jaxpr: Jaxpr, args: Sequence[Any], **params):
  lo_jaxpr = lower_jaxpr2(jaxpr)
  lo_args = [lo_val for aval, x in zip(jaxpr.in_avals, args)
             for lo_val in aval.lower_val(x)]
  lo_outs = prim.bind(*lo_args, call_jaxpr=lo_jaxpr, **params)
  lo_outs_ = iter(lo_outs)
  hi_outs = [t.raise_val(*it.islice(lo_outs_, len(t.lo_ty())))
             for t in jaxpr.out_avals]
  assert next(lo_outs_, None) is None
  return hi_outs

def _eval_jaxpr_to_lojax(prim, *hi_args, call_jaxpr: Jaxpr, **params):
  return _lower_and_eval(prim, call_jaxpr, hi_args, **params)
eval_jaxpr_p.to_lojax = partial(_eval_jaxpr_to_lojax, eval_jaxpr_p)
