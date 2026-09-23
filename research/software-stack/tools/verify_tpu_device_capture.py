#!/usr/bin/env python3
"""Audit TPU XSpace raw fields and numerical captures; never infer from JSON alone."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from xspace_contexts import load_schema, message

SCHEMA = Path('artifacts/jax-stack/xspace-context-audit-002/capture')
REGIONS = ('research_v7_kernel', 'research_v7_dot', 'research_v7_store')


def check(value, why):
    if not value:
        raise ValueError(why)


def profile(path):
    pool = load_schema(SCHEMA)
    space = message(pool, 'tensorflow.profiler.XSpace', path.read_bytes())
    check(not space.errors, 'XSpace collector errors')
    planes = []
    records = []
    for p in space.planes:
        lines = []
        for line in p.lines:
            counts = Counter()
            for event in line.events:
                check(event.metadata_id in p.event_metadata, 'unresolved event metadata')
                name = p.event_metadata[event.metadata_id].name
                counts[name] += 1
                if p.name.startswith('/device:TPU:'):
                    check(event.WhichOneof('data') == 'offset_ps', 'device aggregate is not a timeline event')
                    start = line.timestamp_ns * 1000 + event.offset_ps
                    records.append({'plane': p.name, 'line': line.name, 'name': name,
                                    'start_ps': start, 'duration_ps': event.duration_ps,
                                    'end_ps': start + event.duration_ps})
            if counts:
                lines.append({'name': line.name, 'events': len(line.events), 'names': dict(counts)})
        planes.append({'name': p.name, 'lines': lines})
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size_bytes': path.stat().st_size,
            'warnings': list(space.warnings), 'errors': list(space.errors), 'planes': planes,
            'device_events': records, 'parser': 'raw XSpace with verified pinned protobuf schema',
            'time_unit': 'ps; retain integer line.timestamp_ns * 1000 + offset_ps',
            'counts_complete_for_parsed_file': True, 'hardware_capture_completeness_guaranteed': False}


def compare_regions(default, enabled):
    def select(data, name):
        return [e for e in data['device_events'] if e['line'] == 'XLA TraceMe' and e['name'] == name]
    check(not any(select(default, n) for n in REGIONS), 'default unexpectedly contains regions')
    groups = {n: sorted(select(enabled, n), key=lambda e:e['start_ps']) for n in REGIONS}
    check(all(len(rows) == 5 for rows in groups.values()), 'expected five instances of each named region')
    for i, parent in enumerate(groups[REGIONS[0]]):
        for name in REGIONS[1:]:
            child = groups[name][i]
            check(child['plane'] == parent['plane'] and
                  parent['start_ps'] <= child['start_ps'] <= child['end_ps'] <= parent['end_ps'],
                  'child region escapes kernel or crosses device')
    return {'counts': {n:len(rows) for n,rows in groups.items()}, 'nested_intervals_valid': True}


def selftest(default, enabled):
    rejected = []
    for name, mutate in [
        ('missing-region', lambda d:d['device_events'].pop(next(i for i,e in enumerate(d['device_events']) if e['name'] == REGIONS[1] and e['line'] == 'XLA TraceMe'))),
        ('host-as-device', lambda d:[e.update(line='host') for e in d['device_events'] if e['line'] == 'XLA TraceMe']),
        ('bad-containment', lambda d:[e.update(end_ps=10**30) for e in d['device_events'] if e['name'] == REGIONS[1]]),
    ]:
        bad = copy.deepcopy(enabled)
        mutate(bad)
        try:
            compare_regions(default, bad)
        except ValueError:
            rejected.append(name)
        else:
            raise ValueError('selftest accepted ' + name)
    return rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--enabled', type=Path)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    result = {'evidence_level': 'REPLAY-OFFLINE', 'production_evidence_level': 'RUN-TPU',
              'qualifiers': ['VERSION-SKEW'], 'profiles': {}, 'numerics': []}
    directories = [args.capture] + ([args.enabled] if args.enabled else [])
    for directory in directories:
        inputs = np.load(directory / 'inputs.npz', allow_pickle=False)
        ref = inputs['a'].astype(np.float64) @ inputs['b'].astype(np.float64)
        for p in sorted(directory.rglob('*.npy')):
            actual = np.load(p, allow_pickle=False)
            np.testing.assert_allclose(actual, ref, rtol=2e-5, atol=2e-5)
            result['numerics'].append({'path': str(p), 'max_abs_error':float(np.max(np.abs(actual-ref)))})
        for p in sorted(directory.rglob('*.xplane.pb')):
            result['profiles'][str(p)] = profile(p)
    if args.enabled:
        def named(directory):
            return result['profiles'][str(next((directory / 'named').rglob('*.xplane.pb')))]
        result['region_comparison'] = compare_regions(named(args.capture), named(args.enabled))
        result['negative_tests'] = selftest(named(args.capture), named(args.enabled))
    result['artifacts'] = [{'path': str(p), 'size_bytes':p.stat().st_size,
                          'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                         for d in directories for p in sorted(d.rglob('*')) if p.is_file()]
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'profiles':len(result['profiles']), 'numerical_arrays':len(result['numerics']),
                     'region_comparison':result.get('region_comparison')}))


if __name__ == '__main__':
    main()
