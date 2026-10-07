"""Read pinned component-diagram evidence from Git or a checksum-locked cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

from software_stack_component_hub_data import PINS, REPOSITORIES, SOURCES
from render_jax_internal_stack import SOURCE_SPECS
from render_pallas_inner_outer import SOURCES as PALLAS_SOURCES
from render_overview_software_stack_extensions import EXTRA_ANCHORS


ANCHORS = dict(SOURCES)
ANCHORS.update({name: ('jax', *spec) for name, spec in SOURCE_SPECS.items()})
ANCHORS.update({name: ('jax', *spec) for name, spec in PALLAS_SOURCES.items()})
ANCHORS.update(EXTRA_ANCHORS)
ANCHORS['BufferAssignment']=('xla','xla/service/buffer_assignment.h','class BufferAssignment {')
ANCHORS['Triton compiler']=('xla','xla/backends/gpu/codegen/triton/xtile_compiler.cc','absl::StatusOr<TritonWrapperResult> CompileTritonToLLVM(')
ANCHORS['Triton LLVM translation']=('xla','xla/backends/gpu/codegen/triton/xtile_compiler.cc','TranslateLLVMToLLVMIR(llvm_context.get(), triton_source.module()));')
ANCHORS['GPU custom binary']=('xla','xla/backends/gpu/codegen/cubin_custom_kernel_compiler.cc','CubinCustomKernelCompiler::CompileToTargetBinary(')
ANCHORS['Triton custom thunk']=('xla','xla/backends/gpu/codegen/triton/fusion.cc','return ThunkSequence::Of<CustomKernelThunk>(')


def source_metadata(root, names, fetch=False):
    root = Path(root)
    locked = {p[1].removeprefix('upstream/'): p[3]
              for line in (root / 'upstream-sources.lock').read_text().splitlines()
              if line and not line.startswith('#') and len(p := line.split('|')) >= 4}
    manifest = json.loads((root / 'tools/component_diagram_sources.json').read_text())
    hashes = {(x['repo'], x['path']): x for x in manifest['files']}
    overview = json.loads((root / 'tools/overview_flow_sources.json').read_text())
    for item in overview['files']:
        key = ('xla', item['path'])
        if key in hashes and hashes[key]['sha256'] != item['sha256']:
            raise ValueError(f'Conflicting source manifests: {key}')
        hashes[key] = item
    repos = {ANCHORS[name][0] for name in names}
    checkouts = {}
    for repo in sorted(repos):
        pin = PINS[repo]
        if locked[repo] != pin:
            raise ValueError(f'{repo}: lock changed; review diagram content and source hashes')
        if repo in manifest['repositories'] and manifest['repositories'][repo] != {
                'repository': REPOSITORIES[repo], 'revision': pin}:
            raise ValueError(f'{repo}: diagram source manifest differs from upstream lock')
        if repo == 'xla' and overview['revision'] != pin:
            raise ValueError('Overview source revision differs from upstream lock')
        directory = root / 'upstream' / repo
        checkouts[repo] = (directory / '.git').exists()
        if checkouts[repo]:
            top = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', '--show-toplevel'], text=True).strip()
            head = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
            if Path(top).resolve() != directory.resolve() or head != pin:
                raise ValueError(f'{repo}: checkout does not match the pinned repository')
    files, verified_files, anchors = {}, [], {}
    for name in sorted(names):
        repo, path, needle, *fixed = ANCHORS[name]
        fixed = fixed[0] if fixed else None
        key = (repo, path)
        if key not in files:
            directory = root / 'upstream' / repo
            local = directory / path
            item = hashes.get(key)
            if checkouts[repo]:
                data = subprocess.check_output(['git', '-C', str(directory), 'show', f'{PINS[repo]}:{path}'])
            else:
                if item is None:
                    raise ValueError(f'Missing checksum-locked evidence: {repo}/{path}')
                cached = root / 'artifacts/environment/overview-sources' / PINS[repo] / path
                if not cached.exists() and fetch:
                    url = f'https://raw.githubusercontent.com/{REPOSITORIES[repo]}/{PINS[repo]}/{path}'
                    with urllib.request.urlopen(url, timeout=45) as response:
                        data = response.read()
                    if hashlib.sha256(data).hexdigest() != item['sha256'] or len(data) != item['size_bytes']:
                        raise ValueError(f'Downloaded source checksum mismatch: {repo}/{path}')
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_bytes(data)
                if not cached.exists():
                    raise ValueError(f'Missing pinned source {repo}/{path}; use --fetch-sources to populate the cache')
                data = cached.read_bytes()
            if item and (hashlib.sha256(data).hexdigest() != item['sha256'] or len(data) != item['size_bytes']):
                raise ValueError(f'Pinned source checksum mismatch: {repo}/{path}')
            if local.exists() and local.read_bytes() != data:
                raise ValueError(f'Modified source file: {repo}/{path}')
            files[key] = data.decode().splitlines()
            verified_files.append(dict(repo=repo, path=path, sha256=hashlib.sha256(data).hexdigest()))
        lines = files[key]
        hits = [i + 1 for i, line in enumerate(lines)
                if (line.startswith(needle) if repo == 'jax' and name in (SOURCE_SPECS.keys() | PALLAS_SOURCES.keys()) else needle in line)]
        if fixed is not None:
            hits = [fixed] if fixed in hits else []
        if len(hits) != 1:
            raise ValueError(f'Ambiguous or changed source anchor {name}: {hits}')
        line = hits[0]
        anchors[name] = dict(repo=repo, path=path, line=line, needle=needle,
                             href=f'https://github.com/{REPOSITORIES[repo]}/blob/{PINS[repo]}/{path}#L{line}')
    return dict(source_pins={r: PINS[r] for r in sorted(repos)}, source_anchors=anchors,
                source_files=sorted(verified_files, key=lambda x: (x['repo'], x['path'])),
                evidence='Pinned source inspection and static SVG checks only; no compilation or device execution')
