#!/usr/bin/env python3
"""Recheck source-bound two-CPU propagation and partitioning artifacts."""

import argparse
import copy
import json
from pathlib import Path
import re

import numpy as np

from audit_pass_transitions import native_snapshot
from capture_runtime import verify_binding, verify_current_reader
from cpu_executable_parser import inventory
from cpu_thunk_probe import check_environment_transition
from matmul_probe import ROOT, fingerprint
from pass_events_probe import bind_native
from verify_research import check, read_json

DEFAULT = ROOT / "artifacts/jax-stack/shardy-runtime-001"
SINGLE = ROOT / "artifacts/jax-stack/source-runtime-002/suite/matmul"
ROW = '#sdy.sharding_per_value<[<@mesh, [{"rows"}, {}]>]>'


def record(path):
    return {"path": str(path.relative_to(ROOT)), **fingerprint(path)}


def one(paths):
    paths = list(paths); check(len(paths) == 1, "missing or ambiguous stage artifact")
    return paths[0]


def mlir_snapshot(path):
    from jax._src.interpreters import mlir
    from jaxlib.mlir import ir
    with mlir.make_ir_context():
        module = ir.Module.parse(path.read_text())
        check(module.operation.verify(), "MLIR module failed verification")
        rows = []
        def walk(op):
            op = op.operation
            rows.append({"name": op.name, "attributes": {name: str(op.attributes[name]) for name in op.attributes},
                         "results": [str(v.type) for v in op.results], "operands": [str(v.type) for v in op.operands]})
            for region in op.regions:
                for block in region.blocks:
                    for child in block.operations: walk(child)
        walk(module.operation)
        return rows


def named(rows, name):
    matches = [r for r in rows if r['name'] == name]
    check(len(matches) == 1, "expected one MLIR op: " + name)
    return matches[0]


def check_propagation(stages):
    before, after, exported = [stages[k] for k in ['before', 'after', 'exported']]
    for name in ['stablehlo.dot','stablehlo.add','stablehlo.maximum']:
        a,b,c = [named(rows,name) for rows in (before,after,exported)]
        check(a['results'] == b['results'] == c['results'] == ['tensor<8x12xf32>'], "propagation unexpectedly partitioned shapes")
        check('sdy.sharding' not in a['attributes'] and b['attributes'].get('sdy.sharding') == ROW, "missing new row sharding")
        check(c['attributes'].get('mhlo.sharding') == '"{devices=[2,1]<=[2]}"' and 'sdy.sharding' not in c['attributes'], "final StableHLO export lost sharding")
    rule = named(after,'stablehlo.dot')['attributes']['sdy.sharding_rule']
    check('([i, k], [k, j])->([i, j])' in rule and 'i=8, j=12, k=16' in rule and 'reduction={k}' in rule, "dot factor rule differs")
    check(any(r['name']=='sdy.constant' for r in before) and not any(r['name'].startswith('sdy.') for r in exported), "constant/SDY cleanup differs")


def check_partition(stages):
    before, after = stages['propagated'], stages['partitioned']
    for snap, shape in [(before,'8,12'),(after,'4,12')]:
        dots = [n for n in snap['nodes'].values() if n['opcode']=='kDot']
        check(len(dots)==1 and re.search(r'= f32\['+shape+r'\]',dots[0]['text']), "dot shape differs at partition boundary")
        check('lhs_contracting_dims={1}' in dots[0]['text'] and 'rhs_contracting_dims={0}' in dots[0]['text'], "dot contraction changed")
    check(not any(any(word in op for word in ['allreduce','allgather','reducescatter','collectivepermute','alltoall']) for op in (k.lower().replace('-', '') for k in after['opcode_counts'])), "unexpected collective in row-sharded example")


def check_shards(summary, outputs, reference):
    check(summary['device_count']==2 and summary['process_count']==1 and summary['invocations']==3, "wrong execution scope")
    observed = []
    for shard in summary['output_shards']:
        check(shard['shape']==[4,12], "local result shape differs")
        idx = tuple(slice(*v) for v in shard['index'])
        np.testing.assert_allclose(outputs[f"shard_{shard['device_id']}"], reference[idx], rtol=2e-5, atol=2e-5)
        observed.extend(list(range(8))[idx[0]])
    check(sorted(observed)==list(range(8)), "shard indices overlap or omit rows")


def verify_case(source, mode):
    checked = inventory(source,'RUN-CPU'); manifest=read_json(source/'manifest.json')
    before, after = [read_json(source/name) for name in ['environment-before.json','environment.json']]
    summary = read_json(source/'summary.json')
    check(summary['mode']==mode and summary['jax_use_shardy_partitioner']==(mode=='shardy'), "wrong partitioner mode")
    check(check_environment_transition(before,after)==summary['newly_loaded_native_libraries'], "native mapping change differs")
    binding=verify_binding(source,after,manifest)
    check(bind_native(before,ROOT/manifest['jaxlib_build_manifest'],'absent')==read_json(source/'build-binding-before.json'), "initial native identity differs")
    reader=verify_current_reader(binding)
    with np.load(source/'inputs.npz',allow_pickle=False) as data, np.load(source/'outputs.npz',allow_pickle=False) as outputs:
        check([list(data[k].shape) for k in ['a','w','bias']]==[[8,16],[16,12],[12]], "wrong reference input shapes")
        reference=np.maximum(data['a'].astype(np.float64)@data['w'].astype(np.float64)+data['bias'].astype(np.float64),0)
        errors=[]
        for i in range(3):
            actual=outputs[f'result_{i}'];np.testing.assert_allclose(actual,reference,rtol=2e-5,atol=2e-5)
            errors.append(float(np.max(np.abs(actual-reference))))
        check_shards(summary,outputs,reference)
    check(errors==summary['max_absolute_errors'], "numerical summary differs")
    dump=source/'xla-dump'; prefix='module_0002.jit_row_matmul'
    pass_name='shardy-xla' if mode=='shardy' else 'sharding-propagation'
    paths={'input':source/'stablehlo.mlir',
           'pre_hlo':one(dump.glob(prefix+'.*.before_'+pass_name+'.txt')),
           'propagated':one(dump.glob(prefix+'.*.after_'+pass_name+'.before_spmd-partitioning.txt')),
           'partitioned':one(dump.glob(prefix+'.*.after_spmd-partitioning.*.txt'))}
    hlo={k:native_snapshot(paths[k]) for k in ['pre_hlo','propagated','partitioned']}
    check_partition(hlo)
    mlir_data={'input':mlir_snapshot(paths['input'])}
    if mode=='shardy':
        folder=one((dump/'shardy').glob('module_*.jit_row_matmul'))
        for key,basename in [('roundtrip_input','00.input_module.mlir'),('before','01.before_propagation.mlir'),('after','02.after_propagation.mlir'),('minimal','03.after_minimal_partitioner_with_global_shapes.mlir'),('exported','04.output_module.mlir')]:
            paths[key]=folder/basename;mlir_data[key]=mlir_snapshot(paths[key])
        check_propagation(mlir_data)
        check('mhlo.frontend_attributes' in named(mlir_data['roundtrip_input'],'builtin.module')['attributes'], "missing round-trip frontend payload")
    else:
        check(not (dump/'shardy').exists(), "GSPMD capture unexpectedly has Shardy propagation stages")
        check(not any(r['name'].startswith('sdy.') for r in mlir_data['input']), "GSPMD input unexpectedly has SDY ops")
    # Parse first, then bind all newly loaded MLIR/native reader libraries too.
    reader_after=verify_current_reader(binding)
    return {'mode':mode,'runtime':checked,'build_binding':binding,'reader_before':reader,'reader_after':reader_after,
            'max_absolute_errors':errors,'output_shards':summary['output_shards'],'stage_artifacts':{k:record(p) for k,p in paths.items()},
            'hlo':hlo,'mlir':mlir_data}


def single_partition_reference():
    checked=inventory(SINGLE,'RUN-CPU'); manifest=read_json(SINGLE/'manifest.json')
    binding=verify_binding(SINGLE,read_json(SINGLE/'environment.json'),manifest)
    pairs=[]
    for prefix in ['module_0004.jit_matmul','module_0014.jit_matmul','module_0024.jit_loss','module_0034.jit_batched_loss']:
        before=one((SINGLE/'xla-dump').glob(prefix+'.*.before_shardy-xla.txt'))
        after=one((SINGLE/'xla-dump').glob(prefix+'.*.after_shardy-xla.*.txt'))
        check(before.read_bytes()==after.read_bytes(), "single-partition Shardy boundary changed")
        check('sharding-removal' in before.name and 'xla.sdy.use_tuple_args' not in before.read_text(), "single-partition scope differs")
        pairs.append({'before':record(before),'after':record(after),'identical':True})
    return {'runtime':checked,'build_binding':binding,'pairs':pairs,
            'interpretation':'Paired dump bytes unchanged. Source CPU num_partitions=1 branch passes runSdyShardingPropagation=false; this is not propagation evidence.'}


def selftest(result, source):
    negatives=[]
    def reject(name,fn):
        try:fn()
        except (ValueError,AssertionError,KeyError):negatives.append(name)
        else:raise AssertionError('invalid evidence accepted: '+name)
    for mode in ['hash','missing','level']:
        fake=copy.deepcopy(read_json(source/'shardy/manifest.json'))
        if mode=='hash':fake['artifacts'][0]['sha256']='0'*64
        elif mode=='missing':fake['artifacts'].pop()
        else:fake['evidence_level']='RUN-TPU'
        reject(mode,lambda:inventory(source/'shardy','RUN-CPU',fake))
    original=result['cases'][0]['mlir']
    bad=copy.deepcopy(original);named(bad['after'],'stablehlo.dot')['attributes'].pop('sdy.sharding')
    reject('missing-propagated-dot-sharding',lambda:check_propagation(bad))
    bad2=copy.deepcopy(original);named(bad2['after'],'stablehlo.dot')['results']=['tensor<4x12xf32>']
    reject('propagation-as-partitioning',lambda:check_propagation(bad2))
    bad3=copy.deepcopy(original);named(bad3['after'],'stablehlo.dot')['attributes']['sdy.sharding_rule']='wrong contraction'
    reject('wrong-dot-factor-rule',lambda:check_propagation(bad3))
    bad4=copy.deepcopy(result['cases'][0]['hlo']);bad4['partitioned']=bad4['propagated']
    reject('global-shape-as-local-partition',lambda:check_partition(bad4))
    bad5=copy.deepcopy(result['cases'][0]['hlo']);bad5['partitioned']['opcode_counts']['kAllReduce']=1
    reject('unexpected-collective',lambda:check_partition(bad5))
    reject('missing-stage',lambda:one([]))
    return negatives


def verify(source, run_selftest=False):
    cases=[verify_case(source/mode,mode) for mode in ['shardy','gspmd']]
    check(cases[0]['build_binding']['build_manifest']==cases[1]['build_binding']['build_manifest'], "different builds")
    with np.load(source/'shardy/inputs.npz',allow_pickle=False) as a,np.load(source/'gspmd/inputs.npz',allow_pickle=False) as b:
        check(all(np.array_equal(a[k],b[k]) for k in a.files),"different reference inputs")
    with np.load(source/'shardy/outputs.npz',allow_pickle=False) as a,np.load(source/'gspmd/outputs.npz',allow_pickle=False) as b:
        check(all(np.array_equal(a[k],b[k]) for k in a.files),"partitioner output mismatch")
    result={'evidence_level':'REPLAY-OFFLINE','qualifiers':[],'cases':cases,'single_partition_reference':single_partition_reference(),
            'limits':['Two logical CPUs, one process; not TPU/multi-host/performance evidence.',
                      'Intermediate sharding propagation and later global-to-local partitioning are separately observed.',
                      'Automatic partition search, GSPMD fallback from mixed IR, HloShardingV3 and tuple/alias restoration are source-only or untested.']}
    if run_selftest:result['negative_tests']=selftest(result,source)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=DEFAULT)
    p.add_argument('--result',type=Path);p.add_argument('--selftest',action='store_true');args=p.parse_args()
    result=verify(args.source.resolve(),args.selftest)
    if args.result:args.result.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'cases':len(result['cases']),'artifacts':sum(c['runtime']['artifact_count'] for c in result['cases']),
                      'negative_tests':result.get('negative_tests',[]),'single_partition_pairs':4}))


if __name__=='__main__':main()
