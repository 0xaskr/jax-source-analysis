MemoryKind = str

class RecomputeType: pass
Recompute = RecomputeType()

class SaveableType: pass
Saveable = SaveableType()

class Offloadable(NamedTuple):
  src: MemoryKind
  dst: MemoryKind

RematCases = RecomputeType | SaveableType | Offloadable
RematCases_ = RematCases | bool

def ensure_enum(case: bool | RematCases) -> RematCases:
  if isinstance(case, bool):
    return Saveable if case else Recompute
  if not isinstance(case, (RecomputeType, SaveableType, Offloadable)):
    msg = ("Value returned by a remat policy should be a bool or"
           " `ad_checkpoint.Recompute`, `ad_checkpoint.Saveable` or"
           " `ad_checkpoint.Offloadable(...)`."
           f" Got {case} of type {type(case)}.")
    if isinstance(case, Offloadable):
      msg += ("Did you return `Offloadable` instead of an instantiated"
              " `Offloadable(...)`?")
    raise TypeError(msg)
  return case

# A primitive rule for policy-driven partial evaluation returns a 5-tuple
# with the components representing, respectively:
#  * the JaxprEqn for the 'known' side (or None if there is no known component),
#  * the JaxprEqn for the 'unknown' side (or None),
#  * a list of booleans indicating which of the original outputs are unknown,
#  * a list of booleans indicating which of the original outputs are
#    instantiated (i.e. available) in the 'unknown' side,
#  * a list of Var instances representing residuals to be added (i.e. to be
#    plumbed as outputs of the 'known' side jaxpr and added as input binders to
#    the 'unknown' jaxpr).
PartialEvalCustomResult = tuple[JaxprEqn | None, JaxprEqn | None,
                                Sequence[bool], Sequence[bool], list[Var]]
PartialEvalCustomRule = Callable[
    [Callable[..., RematCases_], Sequence[bool], Sequence[bool], JaxprEqn],
    PartialEvalCustomResult]
partial_eval_jaxpr_custom_rules: dict[Primitive, PartialEvalCustomRule] = {}


ParamsUpdater = Callable[[Sequence[bool], Sequence[bool], Sequence[bool],
                          Sequence[bool], int, dict, dict],
                         tuple[dict, dict]]
ResAvalUpdater = Callable[[dict[str, Any], AbstractValue], AbstractValue]
def _default_res_aval_updater(
    params: dict[str, Any], aval: AbstractValue) -> AbstractValue:
  return aval


def call_partial_eval_custom_rule(
    jaxpr_param_name: str, params_updater: ParamsUpdater,
    saveable: Callable[..., RematCases_], unks_in: list[bool], inst_in: list[bool],
    eqn: JaxprEqn, *, res_aval: ResAvalUpdater = _default_res_aval_updater,
    ctx = contextlib.nullcontext,
  ) -> tuple[JaxprEqn, JaxprEqn, Sequence[bool], Sequence[bool], list[Var]]:
  jaxpr = eqn.params[jaxpr_param_name]
  with ctx(eqn.params):
    jaxpr_known, jaxpr_staged, unks_out, inst_out, num_res = \
        partial_eval_jaxpr_custom(jaxpr, unks_in, inst_in, False, False, saveable)
  ins_known, _ = partition_list(unks_in, eqn.invars)
  out_binders_known, _ = partition_list(unks_out, eqn.outvars)
  _, ins_staged = partition_list(inst_in, eqn.invars)
  _, out_binders_staged = partition_list(inst_out, eqn.outvars)
  params_known = {**eqn.params, jaxpr_param_name: jaxpr_known}
  params_staged = {**eqn.params, jaxpr_param_name: jaxpr_staged}
  params_known, params_staged = params_updater(
      unks_in, inst_in, map(op.not_, unks_out), inst_out, num_res, params_known,
      params_staged)
  residuals = [Var(res_aval(params_known, var.aval))
               for var in jaxpr_staged.invars[:num_res]]
  eqn_known = new_jaxpr_eqn(
      ins_known, [*out_binders_known, *residuals], eqn.primitive, params_known,
      core.eqn_effects(jaxpr_known, ins_known), eqn.source_info, eqn.ctx)
  eqn_staged = new_jaxpr_eqn(
      [*residuals, *ins_staged], out_binders_staged, eqn.primitive,
      params_staged, core.eqn_effects(jaxpr_staged, [*residuals, *ins_staged]),
      eqn.source_info, eqn.ctx)
  assert len(eqn_staged.invars) == len(jaxpr_staged.invars)
  new_inst = [x for x, inst in zip(eqn.invars, inst_in)
              if type(x) is Var and not inst]
  return eqn_known, eqn_staged, unks_out, inst_out, new_inst + residuals

# TODO(mattjj): unify with ParamsUpdater (this one takes an extra int)
ParamsUpdater2 = Callable[[Sequence[bool], Sequence[bool], Sequence[bool],
                           Sequence[bool], int, int, dict, dict],
                          tuple[dict, dict]]

def closed_call_partial_eval_custom_rule(
    jaxpr_param_name: str, params_updater: ParamsUpdater2,
    saveable: Callable[..., RematCases_], unks_in: list[bool], inst_in: list[bool],
    eqn: JaxprEqn, *, res_aval: ResAvalUpdater = _default_res_aval_updater,
  ) -> tuple[JaxprEqn, JaxprEqn, Sequence[bool], Sequence[bool], list[Var]]:
  # TODO(sharadmv,mattjj): dedup this rule with call_partial_eval_custom_rule.
  disallow_output_fwds = tuple(isinstance(v, DropVar) for v in eqn.outvars)
  # TODO(mattjj): this is just for pjit... but let's delete all this code
  from jax._src.sharding_impls import UNSPECIFIED  # pyrefly: ignore[missing-import]
  in_shardings, in_layouts = eqn.params.get('in_shardings'), eqn.params.get('in_layouts')
  if in_shardings is not None:
    assert in_layouts is not None
    disallow_input_fwds = tuple(s is not UNSPECIFIED or l is not None
                                for s, l in zip(in_shardings, in_layouts))
  else:
    disallow_input_fwds = (False,) * len(unks_in)
  jaxpr_known, jaxpr_staged, unks_out, inst_out, num_res_ref, num_res_val, in_fwd, out_fwd = \
      _closed_jaxpr_partial_eval_custom_cached(
          eqn.params[jaxpr_param_name], (*unks_in,), (*inst_in,),
          disallow_input_fwds, disallow_output_fwds, saveable)
  num_res = num_res_ref + num_res_val
  out_binders_known, _ = partition_list(unks_out, eqn.outvars)
  ins_known, _ = partition_list(unks_in, eqn.invars)
  _, ins_staged = partition_list(inst_in, eqn.invars)
  _, out_binders_staged = partition_list(inst_out, eqn.outvars)
  params_known = {**eqn.params, jaxpr_param_name: jaxpr_known}
  params_staged = {**eqn.params, jaxpr_param_name: jaxpr_staged}
  params_known, params_staged = params_updater(
      unks_in, inst_in, map(op.not_, unks_out), inst_out,
      sum(fin is fout is None for fin, fout in zip(in_fwd, out_fwd)),
      num_res, params_known, params_staged)
  res_val_binders, res_ref_binders = split_list(
      [Var(res_aval(params_known, v))
       for v in jaxpr_staged.in_avals[:num_res]], [num_res_val])
  res_val_binders = [v for v, fin, fout in zip(res_val_binders, in_fwd, out_fwd)
                     if fin is fout is None]
  res_val_binders_ = iter(res_val_binders)
  res_val_vars = [out_binders_known[fout] if fout is not None else
                  ins_known[fin] if fin is not None else
                  next(res_val_binders_) for fin, fout in zip(in_fwd, out_fwd)]
  assert next(res_val_binders_, None) is None
  eqn_known = new_jaxpr_eqn(
      [*ins_known, *res_ref_binders], [*out_binders_known, *res_val_binders],
      eqn.primitive, params_known,
      core.eqn_effects(jaxpr_known, [*ins_known, *res_ref_binders]),
      eqn.source_info, eqn.ctx)
  eqn_staged = new_jaxpr_eqn(
      [*res_val_vars, *res_ref_binders, *ins_staged], out_binders_staged,
      eqn.primitive, params_staged,
      core.eqn_effects(jaxpr_staged, [*res_val_vars, *res_ref_binders, *ins_staged]),
      eqn.source_info, eqn.ctx)
  assert len(eqn_staged.invars) == len(jaxpr_staged.in_avals)
  assert len(ins_known) + len(res_ref_binders) == len(jaxpr_known.invars)
  assert len(ins_staged) + len(res_ref_binders) + len(res_val_vars) == len(jaxpr_staged.invars)
  assert len(out_binders_known) + len(res_val_binders) == len(jaxpr_known.outvars)
  new_inst = [x for x, inst in zip(eqn.invars, inst_in)
              if type(x) is Var and not inst]
  new_vars = [*new_inst, *res_val_vars, *res_ref_binders]
  return eqn_known, eqn_staged, unks_out, inst_out, new_vars  # pyrefly: ignore[bad-return]

@weakref_lru_cache
def _closed_jaxpr_partial_eval_custom_cached(
    jaxpr: Jaxpr, unks_in: tuple[bool, ...], inst_in: tuple[bool, ...],
    disallowed_input_forwards: tuple[bool, ...],
    disallowed_output_forwards: tuple[bool, ...],
    saveable: Callable
    ) -> tuple[Jaxpr, Jaxpr, Sequence[bool], Sequence[bool],
               int, int, Sequence[int | None], Sequence[int | None]]:
  jaxpr_known_, jaxpr_staged_, unks_out, inst_out, num_res_val, num_res_ref = \
      partial_eval_jaxpr_stateful(jaxpr, unks_in, inst_in,
                                  False, False, saveable)

  num_out_primals = len(jaxpr_known_.outvars) - num_res_val
  out_vars, res_vars = split_list(jaxpr_known_.outvars, [num_out_primals])

  # Compute which residual value outputs are also primal inputs.
  disallowed, _ = partition_list(unks_in, disallowed_input_forwards)
  idx_map = {id(v): i for i, (v, b) in enumerate(zip(jaxpr_known_.invars, disallowed))
             if not b}
  in_fwd = [idx_map.get(id(v)) for v in res_vars]

  # Compute which residual value outputs are also *undropped* primal outputs.
  disallowed, _ = partition_list(unks_out, disallowed_output_forwards)
  idx_map = {id(v): i for i, (v, b) in enumerate(zip(out_vars, disallowed))
             if not b}
  out_fwd = [idx_map.get(id(v)) for v in res_vars]

  # Prune jaxpr_known_ outputs by removing forwards.
  keep = [f1 is f2 is None for f1, f2 in zip(in_fwd, out_fwd)]
  jaxpr_known_ = prune_jaxpr_outputs(jaxpr_known_, [True] * num_out_primals + keep)

  return (jaxpr_known_, jaxpr_staged_, unks_out, inst_out, num_res_ref,
          num_res_val, in_fwd, out_fwd)


def _jaxpr_forwarding(jaxpr: Jaxpr) -> list[int | None]:
  # Compute which inputs are just forwarded to outputs.
  fwds: dict[Var, Atom] = dict(zip(jaxpr.invars, jaxpr.invars))
  for eqn in jaxpr.eqns:
    if eqn.primitive in forwarding_rules:
      eqn = eqn.replace(invars=[a if type(a) is Literal else fwds.get(a, a)
                                for a in eqn.invars])
      fwd_idx, _ = forwarding_rules[eqn.primitive](eqn)
      for v_orig, idx in zip(eqn.outvars, fwd_idx):
        if idx is not None:
          fwds[v_orig] = eqn.invars[idx]
  idxs: dict[Var, int] = {v: i for i, v in enumerate(jaxpr.invars)}
  return [None if type(v) is Literal else idxs.get(fwds.get(v))  # pyrefly: ignore[bad-argument-type]
          for v in jaxpr.outvars]


def prune_jaxpr_outputs(jaxpr: Jaxpr, used_outputs: Sequence[bool]) -> Jaxpr:
  return _prune_jaxpr_outputs_cached(jaxpr, tuple(used_outputs))

def _prune_jaxpr_outputs(jaxpr: Jaxpr, used_outputs: tuple[bool, ...]) -> Jaxpr:
  outvars = [v for v, b in zip(jaxpr.outvars, used_outputs) if b]
  dbg = core.DebugInfo(
      jaxpr.debug_info.traced_for, jaxpr.debug_info.func_src_info,
      jaxpr.debug_info.arg_names,
      jaxpr.debug_info.filter_result_paths(used_outputs))
  new_jaxpr = jaxpr.replace(outvars=outvars, debug_info=dbg)
  config.enable_checks.value and core.check_jaxpr(new_jaxpr)
  return new_jaxpr
_prune_jaxpr_outputs_cached = weakref_lru_cache(_prune_jaxpr_outputs)

# Since the Jaxpr/Jaxpr merge this is the same as prune_jaxpr_outputs
# (attached consts are preserved by Jaxpr.replace).
prune_closed_jaxpr_outputs = prune_jaxpr_outputs

def dedup_jaxpr_outputs(jaxpr: Jaxpr, num_kept_prefix: int,
                        fwdable_prefix: Sequence[bool] | None = None,
                        ) -> tuple[Jaxpr, list[int | None]]:
  """Prune duplicated outputs, for callers to restore after evaluation."""
  prefix_vars, rest_vars = split_list(jaxpr.outvars, [num_kept_prefix])
  fwdable = ([True] * num_kept_prefix if fwdable_prefix is None
             else fwdable_prefix)
  idx_map = {id(v): i for i, (v, f) in enumerate(zip(prefix_vars, fwdable))
             if f}
  num_kept = num_kept_prefix
  out_fwd: list[int | None] = [None] * num_kept_prefix
  for v in rest_vars:
    if (fwd := idx_map.get(id(v))) is None:
      idx_map[id(v)] = num_kept
      num_kept += 1
    out_fwd.append(fwd)
  jaxpr = prune_jaxpr_outputs(jaxpr, [f is None for f in out_fwd])
  return jaxpr, out_fwd


def dce_jaxpr(jaxpr: Jaxpr, used_outputs: bool | Sequence[bool],
              instantiate: bool | Sequence[bool] = False,
              ) -> tuple[Jaxpr, list[bool]]:
  """Runs dead-code elementation on a given jaxpr.

  Args:
    jaxpr: The jaxpr to DCE.
    used_outputs: A list of bools indicating which outputs are used.
    instantiate: A bool or a list of bools indicating which inputs should be
      considered used, regardless of whether they are actually used in a jaxpr.
      If a bool, the same value is used for all inputs.

  Returns:
    A tuple of ``(new_jaxpr, used_inputs)``.
  """
  if type(used_outputs) is bool:
    used_outputs = (used_outputs,) * len(jaxpr.outvars)
  if type(instantiate) is bool:
    instantiate = (instantiate,) * len(jaxpr.invars)

  return _dce_jaxpr(jaxpr, tuple(used_outputs), tuple(instantiate))


def dce_jaxpr_consts(jaxpr: Jaxpr, used_outputs: Sequence[bool],
                     instantiate: bool | Sequence[bool] = False,
                     ) -> tuple[Jaxpr, list[bool], list[bool]]:
  new_jaxpr, used_inputs_ = dce_jaxpr(convert_constvars_jaxpr(jaxpr),
                                      used_outputs, instantiate)
  used_consts, used_inputs = split_list(used_inputs_, [len(jaxpr.constvars)])
  new_jaxpr = new_jaxpr.with_consts(
      [c for c, used in zip(jaxpr.consts, used_consts) if used])
  return new_jaxpr, used_consts, used_inputs


def _default_dce_rule(
    used_outs: list[bool], eqn: JaxprEqn
  ) -> tuple[list[bool], JaxprEqn | None]:
  if not any(used_outs) and not has_effects(eqn):
    return [False] * len(eqn.invars), None
  return [True] * len(eqn.invars), eqn

dce_rules: dict[Primitive, DCERule] = {}

dceable_effects = effects.EffectTypeSet()

dceable_effects.add_type(ReadEffect)
dceable_effects.add_type(core.NamedAxisEffect)
dceable_effects.add_type(core.InternalMutableArrayEffect)

def _free_ref_dce_rule(
    used_outs: list[bool], eqn: JaxprEqn
) -> tuple[list[bool], JaxprEqn | None]:
  # Never gonna DCE free_ref.
  del used_outs
  return [True] * len(eqn.invars), eqn
dce_rules[core.free_ref_p] = _free_ref_dce_rule


def has_effects(eqn: JaxprEqn) -> bool:
  effs = {e for e in eqn.effects if not dceable_effects.contains(e)}
  return bool(effs)


@weakref_lru_cache
def _dce_jaxpr(jaxpr: Jaxpr, used_outputs: tuple[bool, ...],
               instantiate: tuple[bool, ...]
               ) -> tuple[Jaxpr, list[bool]]:
  env: dict[Var, bool] = {}

  def read(v: Var) -> bool:
    return env.get(v, False)

  def write(x: Atom, b: bool) -> None:
    if type(x) is Var:
      env[x] = read(x) or b

  new_eqns = []
  foreach(write, jaxpr.outvars, used_outputs)
  for eqn in jaxpr.eqns[::-1]:
    used_outs = map(read, eqn.outvars)
    rule = dce_rules.get(eqn.primitive, _default_dce_rule)
    used_ins, new_eqn = rule(used_outs, eqn)
    if new_eqn is not None:
      new_eqns.append(new_eqn)
    foreach(write, eqn.invars, used_ins)
  used_inputs = map(read, jaxpr.invars)
  used_inputs = map(op.or_, instantiate, used_inputs)

  invars = [v for v, b in zip(jaxpr.invars, used_inputs)   if b]
  outvars = [v for v, b in zip(jaxpr.outvars, used_outputs) if b]
  eqns = new_eqns[::-1]
  jaxpr_effects = make_jaxpr_effects(jaxpr.constvars, invars, outvars, eqns)

  dbg = core.DebugInfo(
      jaxpr.debug_info.traced_for, jaxpr.debug_info.func_src_info,
      jaxpr.debug_info.filter_arg_names(used_inputs),
      jaxpr.debug_info.filter_result_paths(used_outputs))
  new_jaxpr = jaxpr.replace(invars=invars, outvars=outvars, eqns=eqns,
                            effects=jaxpr_effects, debug_info=dbg)
  config.enable_checks.value and core.check_jaxpr(new_jaxpr)

  return new_jaxpr, used_inputs

DCERule = Callable[[list[bool], JaxprEqn],
                   tuple[list[bool], JaxprEqn | None]]


@weakref_lru_cache
def _cached_closed_call_dce(jaxpr_, used_outputs: tuple[bool, ...]
                            ) -> tuple[Jaxpr, list[bool]]:
  # dce_jaxpr preserves attached consts (constvars are never pruned).
  return dce_jaxpr(jaxpr_, used_outputs)

def dce_jaxpr_closed_call_rule(used_outputs: list[bool], eqn: JaxprEqn
                               ) -> tuple[list[bool], JaxprEqn | None]:
  # TODO(mattjj): de-duplicate with above rule?
  if not any(used_outputs) and not has_effects(eqn):
    return [False] * len(eqn.invars), None
  jaxpr_ = eqn.params['call_jaxpr']
  closed_jaxpr, used_inputs = _cached_closed_call_dce(jaxpr_, tuple(used_outputs))
  new_invars = [v for v, used in zip(eqn.invars, used_inputs) if used]
  effects = core.eqn_effects(closed_jaxpr, new_invars)
  new_params = dict(eqn.params, call_jaxpr=closed_jaxpr)
  new_eqn = new_jaxpr_eqn(
      new_invars,
      [v for v, used in zip(eqn.outvars, used_outputs) if used],
      eqn.primitive, new_params, effects, eqn.source_info, eqn.ctx)
  return used_inputs, new_eqn

def close_jaxpr(jaxpr: Jaxpr) -> Jaxpr:
  # Now that Jaxpr and ClosedJaxpr are merged, every jaxpr is closed: the
  # constvars are exactly the inputs with attached const values.
  return jaxpr

def move_binders_to_front(closed_jaxpr: Jaxpr, to_move: Sequence[bool]
                          ) -> Jaxpr:
  return _move_binders_to_front(closed_jaxpr, tuple(to_move))

@weakref_lru_cache
def _move_binders_to_front(jaxpr: Jaxpr, to_move: tuple[bool, ...]
                           ) -> Jaxpr:
  assert len(jaxpr.in_avals) == len(to_move)
  new_invars = _move_to_front(jaxpr.invars, to_move)
  if jaxpr.debug_info.arg_names is None:
    new_arg_names = None
  else:
    new_arg_names = tuple(_move_to_front(jaxpr.debug_info.arg_names, to_move))
  dbg = jaxpr.debug_info._replace(arg_names=new_arg_names)
  return jaxpr.replace(invars=new_invars, debug_info=dbg)

def _move_to_front(lst: Sequence, to_move: Sequence[bool]) -> Sequence:
  return ([elt for elt, move in zip(lst, to_move) if move] +
          [elt for elt, move in zip(lst, to_move) if not move])

def move_binders_to_back(closed_jaxpr: Jaxpr, to_move: Sequence[bool]
                         ) -> Jaxpr:
  assert len(to_move) <= len(closed_jaxpr.invars)
  to_move = [*to_move] + [False] * (len(closed_jaxpr.invars) - len(to_move))
  return move_binders_to_front(closed_jaxpr, map(op.not_, to_move))


class DynamicJaxprTracer(Tracer['DynamicJaxprTrace']):
  __slots__ = ['val', 'parent', '_debug_info']

  _trace: DynamicJaxprTrace

  def __init__(self, trace: DynamicJaxprTrace,
               aval: core.AbstractValue,
               val : Atom,
               line_info: source_info_util.SourceInfo | None = None,
               parent : TracingEqn | None = None):
    # TODO(dougalm): Remove aval. It's redundant now that we have val.
    Tracer.__init__(self, trace, aval)  # slightly faster than super()
    self._line_info = line_info
    self._debug_info = self._trace.frame.debug_info  # for UnexpectedTracerError
    self.val = val
    self.parent = parent

  def _short_repr(self):
    return f"JitTracer({self.aval})"

  def full_lower(self):
    atom = self.val
    if isinstance(atom, Literal):
      return atom.val
    else:
      maybe_const = self._trace.frame.constvar_to_val.get(atom)
      if maybe_const is None:
        return self
      else:
        return core.full_lower(maybe_const)

  def _contents(self):
    return ()

  def _origin_msg(self):
    invar_pos, progenitor_eqns = self._trace.frame.find_progenitors(self)
    dbg = self._debug_info
    if dbg is None:
      return ""

    origin = ("The error occurred while tracing the function "
              f"{dbg.func_src_info} for {dbg.traced_for}. ")
    if invar_pos:
      try:
        arg_names = [(dbg.arg_names[i] if dbg.arg_names is not None else "unknown")
                     for i in invar_pos]
      except IndexError:
        return ""  # TODO(mattjj): figure out when not (invar_pos < len(arg_info))
      if len(arg_names) == 1:
        arg_info_str = f"the argument {arg_names[0]}"
      elif len(arg_names) == 2:
        arg_info_str = f"the arguments {arg_names[0]} and {arg_names[1]}"
      else:
        *rest, last = arg_names
        arg_info_str = f"the arguments {', '.join(rest)}, and {last}"
      origin += ("This concrete value was not available in Python because it "
                 f"depends on the value{'s' if len(invar_pos) > 1 else ''} "
                 f"of {arg_info_str}.")
    elif progenitor_eqns:
      msts = ["  operation "
              f"{core.pp_eqn(eqn, core.JaxprPpContext(), core.JaxprPpSettings(print_shapes=True))}\n"
              f"    from line {source_info_util.summarize(eqn.source_info)}"
              for eqn in progenitor_eqns[:5]]  # show at most 5
      origin += ("This value became a tracer due to JAX operations on these lines:"
                 "\n\n" + "\n\n".join(msts))
      if len(progenitor_eqns) > 5:
        origin += "\n\n(Additional originating lines are not shown.)"
    return "\n" + origin

  def get_const(self):
    return self._trace.get_const(self)

  def get_referent(self):
    frame = self._trace.frame
    atom = self.val
    val = frame.constvar_to_val.get(atom) if isinstance(atom, Var) else None
    return self if val is None else get_referent(val)

