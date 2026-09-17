core.pytype_aval_mappings[DynamicJaxprTracer] = lambda x: x.aval

def make_jaxpr_effects(constvars, invars, outvars, eqns) -> effects.Effects:
  jaxpr_effects = set()
  input_vars = {*constvars, *invars}
  mut_arrays = set()
  for eqn in eqns:
    if eqn.primitive.ref_allocating:
      outvar, = eqn.outvars
      mut_arrays.add(outvar)
    for eff in eqn.effects:
      if isinstance(eff, effects.JaxprInputEffect):
        if eff.input in mut_arrays:
          continue
        if eff.input not in input_vars:
          # TODO(mattjj): ask for forgiveness
          dbg = type('Fake', (), {'resolve_result_paths': lambda self_: self_,
                                  'assert_arg_names': lambda _, __: None,
                                  'assert_result_paths': lambda _, __: None,
                                  })()
          raise ValueError(
                f"`JaxprInputEffect` {eff} does not have "
                f"a corresponding jaxpr input."
                f"\n Equation: {eqn}\n"
                f"\n Effects: {eqn.effects}\n"
                "\n Jaxpr: "
                f"{core.Jaxpr(constvars, invars, outvars, eqns, set(), dbg)}")
      jaxpr_effects.add(eff)
  return jaxpr_effects


class JaxprStackFrame:
  __slots__ = (
      'gensym', 'constid_to_tracer', 'constvar_to_val', 'tracing_eqns',
      'invars', 'effects', 'debug_info', 'is_high', 'auto_dce')

  gensym: Callable[[AbstractValue], Var]
  constid_to_tracer: WeakValueDictionary[ConstId, DynamicJaxprTracer]
  constvar_to_val: dict[Var, Any]
  tracing_eqns: list[ReferenceType[TracingEqn] | TracingEqn]
  invars: list[Var]
  effects: core.Effects
  debug_info: core.DebugInfo | None
  is_high: bool
  auto_dce: bool

  def __init__(self, debug_info: core.DebugInfo | None, auto_dce: bool):
    self.gensym = core.gensym()
    self.constid_to_tracer = WeakValueDictionary()
    self.constvar_to_val = {}
    self.tracing_eqns = []      # cleared when we pop frame from main
    self.invars = []
    self.effects = set()
    self.debug_info = debug_info
    self.is_high = False
    self.auto_dce = auto_dce

  def add_eqn(self, eqn: TracingEqn):
    assert isinstance(eqn, TracingEqn)
    r = eqn if (eqn.effects or not self.auto_dce) else ref(eqn)
    self.tracing_eqns.append(r)

  def get_eqns(self):
    eqns = []
    for tracing_eqn in self.tracing_eqns:
      e = tracing_eqn() if isinstance(tracing_eqn, ReferenceType) else tracing_eqn
      if e is None: continue
      if isinstance(e, TracingEqn):
        e = JaxprEqn(
            [t.val for t in e.in_tracers],
            e.outvars, e.primitive, e.params, e.effects, e.source_info, e.ctx)
      eqns.append(e)
    return eqns

  def to_jaxpr(
      self, trace: DynamicJaxprTrace,
      out_tracers: Sequence[Tracer],
      debug_info: core.DebugInfo,
      source_info: SourceInfo,
    ) -> tuple[Jaxpr, list[Any]]:
    # We are careful to snapshot constvar_to_val before we call get_eqns(),
    # to avoid the following scenario:
    # * we call get_eqns(), snapshotting the equations.
    # * garbage collection runs, deleting some TracingEqns, which may
    #   transitively free constant Vars.
    # * we now have equations with dangling Var references.
    # If we snapshot the Vars first we won't have this problem.
    constvars, constvals = unzip2(self.constvar_to_val.copy().items())
    eqns = self.get_eqns()
    outvars = [t.val for t in out_tracers]
    effs = make_jaxpr_effects(constvars, self.invars, outvars, eqns)

    all_vars = it.chain(constvars, self.invars, outvars)
    is_high = self.is_high or any(v.aval.is_high for v in all_vars)

    jaxpr = Jaxpr(constvars, self.invars, outvars, eqns, effs, debug_info, is_high)
    return jaxpr, list(constvals)

  def newvar(self, aval):
    return self.gensym(aval)

  def find_progenitors(self, tracer):
    eqns = self.get_eqns()
    var = tracer.val
    if not var or isinstance(var, Literal):
      return None, None
    active_vars = {var}
    for eqn in eqns[::-1]:
      produced = set(eqn.outvars) & active_vars
      if produced:
        active_vars.difference_update(produced)
        active_vars.update({v for v in eqn.invars if type(v) is Var})
    invar_positions = [i for i, v in enumerate(self.invars) if v in active_vars]
    constvars = active_vars & set(self.constvar_to_val.copy())
    const_eqns = [eqn for eqn in eqns if any(
        v in constvars if type(v) is Var else type(v) is Literal
        for v in eqn.invars)]
    return invar_positions, const_eqns


ConstFoldRule = Callable[
    [list[Any | None], Any, list[AbstractValue]],
    list[Any | None] | None,
]
const_fold_rules: dict[Primitive, ConstFoldRule] = {}

ForwardingRule = Callable[
    [JaxprEqn],
    tuple[list[int | None], JaxprEqn | None]
]
forwarding_rules: dict[Primitive, ForwardingRule] = {}


@multi_weakref_lru_cache
def _cached_abstract_eval(primitive: core.Primitive, *avals, **params):
  return primitive.abstract_eval(*avals, **params)


def _verify_params_are_hashable(
    primitive: core.Primitive, params: dict[str, Any]) -> None:
  for k, v in params.items():
    try:
      hash(v)
    except TypeError as e:
      raise TypeError(
        "As of JAX v0.7, parameters to jaxpr equations must have __hash__ and "
        f"__eq__ methods. In a call to primitive {primitive}, the value of "
        f"parameter {k} was not hashable: {v}") from e

# We use TracingEqn instead JaxprEqn during tracing to allow automatic
# on-the-fly DCE based on Python refcounting. DynamicJaxprTracers point to
# TracingEqns which point to DynamicJaxprTracers and unreachable constants can
# be freed.

@dataclass(slots=True, weakref_slot=True)
class TracingEqn:
  in_tracers: list[DynamicJaxprTracer]
  outvars: list[Var]
  primitive: Primitive
  params: dict[str, Any]
  effects: core.Effects
  source_info: source_info_util.SourceInfo
  ctx: JaxprEqnContext

  def __init__(self, in_tracers, outvars, primitive, params, effects, source_info, ctx):
    self.in_tracers = in_tracers
    self.outvars = outvars
    self.primitive = primitive
    self.params = params
    self.effects = effects
    self.source_info = source_info
    self.ctx = ctx

  # Allow TracingEqn to duck-type JaxpeEqn because some of the forwarding
  # rules need to work with both. TODO(dougalm): remove this once we fix
  # forwarding.
  @property
  def invars(self):
    return self.in_tracers

class DynamicJaxprTrace(core.Trace):
  __slots__ = ("frame", "tag", "parent_trace")

  # Note that tag is only used when DynamicJaxprTrace is associated with a LinearizeTrace;
  # otherwise it will be undefined.
  tag: core.TraceTag
  frame: JaxprStackFrame
  parent_trace: core.Trace | None
  requires_low: bool

  def __init__(self, debug_info: core.DebugInfo | None,
               parent_trace: core.Trace | None = None,
               lower: bool = False, auto_dce: bool =False):
    super().__init__()
    self.requires_low = lower
    self.frame = JaxprStackFrame(debug_info, auto_dce)
    self.parent_trace = parent_trace

  def invalidate(self):
    # TODO(mattjj): exposed existing tracer leaks; fix them and re-enable!
    # super().invalidate()

    # avoid cyclic refs
    self.frame.tracing_eqns = []  # thunk -> eqn -> in_tracers -> trace ->
    # -> frame -> tracing_eqns -> thunk

    # TODO(dougalm): we might be able to remove these given refcounting dce
    self.frame.constid_to_tracer = {}  # pyrefly: ignore[bad-assignment]
    self.frame.constvar_to_val = {}

  def to_jaxpr_tracer(self, x, source_info: SourceInfo):
    if isinstance(x, DynamicJaxprTracer) and x._trace is self:
      return x
    else:
      m = getattr(x, "dimension_as_value", None)
      if m is not None:
        with core.set_current_trace(self):
          x = m()
        return self.to_jaxpr_tracer(x, source_info)
      else:
        return self.new_const(x, source_info)

  def var_to_tracer(self, var, source_info, parent=None):
    return DynamicJaxprTracer(self, var.aval, var, source_info, parent)

  def new_arg(self, aval, source_info: SourceInfo):
    var = self.frame.newvar(aval)
    tracer = DynamicJaxprTracer(self, aval, var, source_info)
    self.frame.invars.append(var)
    return tracer

  def make_eqn(self, in_tracers, out_avals, primitive, params,
               effects, source_info=None, ctx = None):
    source_info = source_info or source_info_util.new_source_info()
    ctx = ctx or core.current_jaxpr_eqn_context()
    outvars = map(self.frame.newvar, out_avals)
    if effects:
      effects = core.resolve_input_effects(effects, [t.val for t in in_tracers])
    if config.enable_checks.value:
      assert all(isinstance(x, DynamicJaxprTracer) for x in in_tracers)
      assert all(isinstance(v,  Var)               for v in outvars)
    eqn = TracingEqn(in_tracers, outvars, primitive, params, effects, source_info, ctx)
    out_tracers = [self.var_to_tracer(v, source_info, eqn) for v in outvars]
    return eqn, out_tracers

  def emit_eqn(self, in_tracers, out_avals, primitive, params, effects, source_info=None, ctx=None):
    eqn, out_tracers = self.make_eqn(in_tracers, out_avals, primitive, params, effects, source_info, ctx)
    self.frame.add_eqn(eqn)
    return out_tracers

  def new_const(self, c, source_info: SourceInfo,
                aval: AbstractValue | None = None):
    # TODO(mattjj): for ints, or hashable consts, don't rely on id
    tracer = self.frame.constid_to_tracer.get(id(c))
    if tracer is None:
      if aval is None:
        aval = typeof(c)
      tracer = self._new_const(aval, c, source_info)
    return tracer

  pure = lift = new_const

  def _new_const(self, aval, c, source_info: SourceInfo) -> DynamicJaxprTracer:
    id_c = id(c)
    assert type(c) not in (int, float, complex, np.generic, np.ndarray), (
        f"non-canonical constant of type {type(c).__name__}: {c!r}")
    if core.is_literalable(c, self.frame.auto_dce):
      val = Literal(c, aval)
      return DynamicJaxprTracer(self, aval, val, source_info)
    else:
      var = self.frame.newvar(aval)
      tracer = DynamicJaxprTracer(self, aval, var, source_info)
      self.frame.constid_to_tracer[id_c] = tracer
      self.frame.constvar_to_val[var] = c
      finalize(tracer, self.finalize_const, var, id_c)
      return tracer

  def finalize_const(self, var, constid):
    self.frame.constvar_to_val.pop(var, None)

  def get_const(self, tracer) -> Any:
    atom = tracer.val
    if isinstance(atom, Literal):
      return atom.val
    else:
      return self.frame.constvar_to_val.get(atom)

  def stage_value(self, val):
    if config.eager_constant_folding.value and not isinstance(val, Tracer):
      return core.eval_trace.stage_value(val)
    source_info = source_info_util.current()
    return self.to_jaxpr_tracer(val, source_info=source_info)

  def process_primitive(self, primitive, tracers, params, /):
    self.frame.is_high |= primitive.is_high(*map(typeof, tracers), **params)
    if config.eager_constant_folding.value and not any(isinstance(x, Tracer) for x in tracers):
      avals = tuple(core.typeof(x) for x in tracers)
      return primitive.bind_with_trace(core.eval_trace, tracers, avals, params)
    source_info = source_info_util.current()
    to_jaxpr_tracer = partial(self.to_jaxpr_tracer, source_info=source_info)
    jaxpr_tracers = map(to_jaxpr_tracer, tracers)
    if primitive in custom_staging_rules:
      return custom_staging_rules[primitive](self, source_info, *jaxpr_tracers,
                                             **params)
    return self.default_process_primitive(
        primitive, jaxpr_tracers, params, source_info)

  def default_process_primitive(self, primitive, tracers, params,
                                source_info=None):
    avals = [t.aval for t in tracers]
    # TODO(mattjj): make custom_lin have hashable params.
    # TODO(dougalm): add an attribute to primitives to mark primitives with
    # effectful abstract_eval rules.
    if (primitive.ref_allocating or
        primitive.name in ("custom_lin", "call_hi_primitive_linearized",
                           "call_hi_primitive")):
      out_avals, effs = primitive.abstract_eval(*avals, **params)
    else:
      try:
        out_avals, effs = _cached_abstract_eval(primitive, *avals, **params)
      except Exception:
        # TODO(phawkins): remove this 3 months after the release of JAX v0.7.
        _verify_params_are_hashable(primitive, params)
        raise

    if isinstance(out_avals, (tuple, list)) != primitive.multiple_results:
      raise ValueError(f"{primitive}.abstract_eval() method should return "
                       f"a tuple or a list iff {primitive}.multiple_results.")
    out_avals = [out_avals] if not primitive.multiple_results else out_avals
    source_info = source_info or source_info_util.current()

    maybe_consts_out = try_constant_folding(primitive, tracers, params, out_avals)
    if maybe_consts_out is not None:
      eqn = None
      out_tracers = [self.new_const(c, source_info=source_info, aval=aval)
                     for c, aval in zip(maybe_consts_out, out_avals)]
    else:
      eqn, out_tracers = self.make_eqn(tracers, out_avals, primitive, params,
                                       effs, source_info=source_info)
    # Input-to-output tracer forwarding
    no_input_effects = not any(isinstance(e, effects.JaxprInputEffect) for e in effs)
    if eqn is not None and no_input_effects and primitive in forwarding_rules:
      in_fwd, eqn = forwarding_rules[primitive](eqn)
      for out_idx, in_idx in enumerate(in_fwd):
        if in_idx is not None:
          out_tracers[out_idx] = tracers[in_idx]

    if eqn is not None:
      self.frame.add_eqn(eqn)  # pyrefly: ignore[bad-argument-type]
    return out_tracers if primitive.multiple_results else out_tracers.pop()

  def process_custom_jvp_call(self, prim, fun: lu.WrappedFun,
                              jvp: lu.WrappedFun, tracers, /, *,
                              symbolic_zeros: bool):
    if self.requires_low:
      with core.set_current_trace(self):
        return fun.call_wrapped(*tracers)

    if config.eager_constant_folding.value and not any(isinstance(x, Tracer) for x in tracers):
      avals = tuple(core.typeof(x) for x in tracers)
      return prim.bind_with_trace(
        core.eval_trace, tracers, avals,
        dict(subfuns=(fun, jvp), symbolic_zeros=symbolic_zeros))
    source_info = source_info_util.current()
    to_jaxpr_tracer = partial(self.to_jaxpr_tracer, source_info=source_info)
    tracers = map(to_jaxpr_tracer, tracers)
    in_avals = [t.aval for t in tracers]
    in_tangent_avals = [t.to_tangent_aval() for t in in_avals]
    fun_jaxpr, out_avals, consts = trace_to_jaxpr_dynamic(fun, in_avals, lower=self.requires_low)
    self.frame.is_high |= fun_jaxpr.is_high
    closed_fun_jaxpr = convert_constvars_jaxpr(fun_jaxpr)

    @partial(lu.wrap_init, debug_info=jvp.debug_info)
    @_memoize
    def jvp_jaxpr_thunk(*in_zeros):
      for store in jvp.stores: store and store.reset()
      nz_tangent_avals, zero_avals = partition_list(in_zeros, in_tangent_avals)
      jvp_, out_zeros = _jvp_jaxpr_zeros(jvp, in_zeros, tuple(zero_avals))
      in_avals_ = (*in_avals, *nz_tangent_avals)
      jaxpr, _, out_consts = trace_to_jaxpr_dynamic(jvp_.with_unknown_names(),
                                                    in_avals_, lower=self.requires_low)
      return jaxpr, out_consts, out_zeros()

    const_tracers = map(to_jaxpr_tracer, consts)
    return self.emit_eqn(
        [*const_tracers, *tracers], out_avals, prim,
        dict(call_jaxpr=closed_fun_jaxpr,
             jvp_jaxpr_fun=jvp_jaxpr_thunk,
             num_consts=len(consts),
             symbolic_zeros=symbolic_zeros),
        core.positional_effects(closed_fun_jaxpr),
        source_info=source_info)

  def process_custom_vjp_call(self, prim: core.Primitive,
                              fun: lu.WrappedFun,
                              fwd: lu.WrappedFun, bwd: lu.WrappedFun, tracers, /, *,
                              out_trees: Callable[[], tuple[PyTreeDef, PyTreeDef, list[int | None]]],
                              symbolic_zeros: bool):
    if self.requires_low:
      with core.set_current_trace(self):
        return fun.call_wrapped(*tracers)

    if config.eager_constant_folding.value and not any(isinstance(x, Tracer) for x in tracers):
      avals = tuple(core.typeof(x) for x in tracers)
      return prim.bind_with_trace(
        core.eval_trace, tuple(tracers), avals,
        dict(subfuns=(fun, fwd, bwd), out_trees=out_trees,
             symbolic_zeros=symbolic_zeros))
    source_info = source_info_util.current()
    to_jaxpr_tracer = partial(self.to_jaxpr_tracer, source_info=source_info)
    tracers = map(to_jaxpr_tracer, tracers)
    in_avals = [t.aval for t in tracers]
    fun_jaxpr, out_avals, consts = trace_to_jaxpr_dynamic(fun.with_unknown_names(), in_avals, lower=self.requires_low)
    self.frame.is_high |= fun_jaxpr.is_high
    num_consts = len(consts)
    closed_fun_jaxpr = convert_constvars_jaxpr(fun_jaxpr)

    @partial(lu.wrap_init, debug_info=fwd.debug_info)
    @_memoize
    def fwd_jaxpr_from_zeros(*zeros):
      for store in fwd.stores: store and store.reset()
      fwd_ = _interleave_fun(fwd.with_unknown_names(), zeros)
      jaxpr, _, consts = trace_to_jaxpr_dynamic(fwd_, in_avals, lower=self.requires_low)
      return jaxpr, consts

    def out_trees_():
      out_tree, res_tree, input_fwds = out_trees()
      input_fwds = [f if f is None else f + num_consts for f in input_fwds]
      return out_tree, res_tree, input_fwds

    const_tracers = map(to_jaxpr_tracer, consts)
    return self.emit_eqn(
        [*const_tracers, *tracers], out_avals, prim,
        dict(call_jaxpr=closed_fun_jaxpr,
             fwd_jaxpr_thunk=fwd_jaxpr_from_zeros,
             num_consts=num_consts,
             bwd=bwd, out_trees=out_trees_,
             symbolic_zeros=symbolic_zeros),
        core.positional_effects(closed_fun_jaxpr),
        source_info=source_info)

  def to_jaxpr(self, out_tracers: Sequence[Tracer],
               debug_info: core.DebugInfo, source_info: SourceInfo):
    return self.frame.to_jaxpr(self, out_tracers, debug_info, source_info)


custom_staging_rules: dict[Primitive, Callable] = {}

@lu.transformation2
def _interleave_fun(f, every_others, *args, **kwargs):
  args_ = [x for pair in zip(args, every_others) for x in pair]
  return f(*args_, **kwargs)

# TODO: consider renaming to "lazy_thunk"
