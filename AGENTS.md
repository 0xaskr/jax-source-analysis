# JAX source workspace

This workspace retains the pinned upstream sources, development environment,
and reusable tool and lab source code. Previous analysis documents, captures,
plans, and progress records have been removed at the user's request.

Before working:

1. Inspect the root repository and relevant source trees with `git status`.
2. Use `upstream-sources.lock`, `source-archives.lock`, and the environment locks
   as the source of truth for revisions and dependencies.
3. Follow the current user task. Do not resume or recreate the former analysis
   queue unless requested.

Preserve upstream sources, source archives, local code, and the development
environments during cleanup. Keep upstream patches intentional and reversible.
Inspect pinned local source before relying on changing online documentation.

Keep source inspection, CPU execution, TPU simulation, offline TPU compilation,
and real TPU execution distinct. Record `VERSION-SKEW` when runtime binaries do
not match the cited sources. Never call Mosaic TPU MLIR "LLO".

For environment changes, run `python3 -B tools/sync-environment.py check` and the
corresponding selftests. Analysis validators remain available in `tools/` for
future evidence bundles; the former status and coverage files are absent.
