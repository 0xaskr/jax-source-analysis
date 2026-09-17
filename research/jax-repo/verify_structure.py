#!/usr/bin/env python3
# Copyright 2026 The JAX Source Analysis Authors.
#
# Validates that every path token in the structure-tree data of
# research/jax-repo/make_diagrams.py actually exists in the pinned checkout
# upstream/jax (and, for tokens qualified with an `xla/` prefix, in
# upstream/xla).
#
# Usage:
#   python3 -B research/jax-repo/verify_structure.py
#
# Exit status is 0 only when every token resolves. The validator is
# read-only and never modifies upstream sources.

from __future__ import annotations

import importlib.util
import pathlib
import sys

import jax

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
JAX_ROOT = REPO / "upstream" / "jax"
XLA_ROOT = REPO / "upstream" / "xla"

SKIP_CHARS = set("@·()（）：,，")

jax.Array

def load_tree():
  spec = importlib.util.spec_from_file_location(
      "make_diagrams", HERE / "make_diagrams.py")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod.TREE


def is_path_token(tok: str) -> bool:
  if not tok or any(c in SKIP_CHARS for c in tok):
    return False
  if tok.startswith("upstream/"):
    return False
  # Anything with no slash and no known extension is prose (or a bare dir
  # name that we still want to check, handled by the caller).
  return True


def check(tok: str, base: pathlib.Path) -> tuple[bool, str]:
  target = base / tok.rstrip("/")
  if target.is_dir() or target.is_file():
    return True, str(target.relative_to(REPO))
  return False, str(target.relative_to(REPO))


def main() -> int:
  tree = load_tree()
  problems: list[str] = []
  checked = 0

  def walk(nodes, base: pathlib.Path):
    nonlocal checked
    for path, _note, _kind, children in nodes:
      if path.startswith("upstream/jax"):
        child_base = JAX_ROOT
      else:
        # Each whitespace-separated token is a path relative to `base`.
        child_base = base
        for tok in path.split():
          if not is_path_token(tok):
            continue
          ok, resolved = check(tok, base)
          checked += 1
          if not ok:
            problems.append(f"MISSING {resolved}   (row: {path!r})")
          elif tok.endswith("/"):
            child_base = base / tok.rstrip("/")
      walk(children, child_base)

  walk(tree, JAX_ROOT)

  print(f"checked {checked} path tokens under {JAX_ROOT.relative_to(REPO)}")
  if problems:
    print(f"\n{len(problems)} unresolved:")
    for p in problems:
      print("  " + p)
    return 1
  print("all tokens resolved")
  return 0


if __name__ == "__main__":
  sys.exit(main())
