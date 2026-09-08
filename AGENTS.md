# Long-running JAX source analysis

This repository is a persistent, multi-session investigation of the pinned JAX
software stack from the public Python API through TPU-specific LLO and hardware.
Tokamax and the private framework are workload providers only; their layers above
JAX are outside the analysis boundary.

Before starting work:

1. Read `HANDOFF.md`, `PLAN.md`, `manifests/status.json`,
   `manifests/baseline.json`, and `manifests/coverage.json`.
2. Run `.venv/bin/python tools/project-status.py --check`, then
   `.venv/bin/python tools/project-status.py` for the recovery summary. Inspect the
   root repository plus every source tree that the task may modify with
   `git status`.
3. Treat `status.next_actions` as the recovery queue and continue the
   `first-ready-action` printed by `project-status.py`. Preserve completed work
   across context compaction and sessions.

Use the revisions in `upstream-sources.lock` and the repository manifests as the
source of truth. Inspect the pinned local source before consulting changing online
documentation. Keep the regular JAX/XLA TPU path separate from the Pallas/Mosaic
TPU path, and never call Mosaic TPU MLIR "LLO".

Every technical claim must follow `docs/contributing/evidence-conventions.md`.
CPU execution, TPU simulation, offline TPU compilation, TPU execution, and source
inspection are different evidence levels. Do not infer TPU runtime, LLO, hardware,
communication, memory, or performance behavior from CPU-only results. Add
`VERSION-SKEW` whenever runtime binaries do not match the cited source revision.

Prefer small runnable probes, captured IR/logs, machine-readable manifests,
source indexes, explicit invariants, and reversible patches. Keep large or private
artifacts out of Git and record sanitized locators plus hashes. Do not leave pinned
upstream trees dirty unless the change is intentional, recorded by a reversible
patch, and reflected in status.

After each independently reviewable major milestone:

1. Run the checks appropriate to the changed scope.
   Evidence or coverage changes include the corresponding semantic validator and
   selftest; project-state changes include `tools/selftest-project-status.py`.
2. Update `PLAN.md` and `manifests/status.json` with results and the next command.
3. Create a focused commit and push it to the current GitHub remote. If the push is
   blocked, record the exact remote or authentication problem and resume the same
   milestone after it is repaired.
