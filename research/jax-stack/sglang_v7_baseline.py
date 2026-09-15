#!/usr/bin/env python3
"""Capture real Qwen3 inference using unchanged SGLang benchmark entry functions."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import time


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')


def main():
    import jax
    import numpy as np
    from huggingface_hub import snapshot_download
    from sgl_jax import bench_one_batch as bench
    from sgl_jax.srt.server_args import ServerArgs
    from sgl_jax.srt.entrypoints.engine import _set_envs_and_config

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument('--research-output', type=Path, required=True)
    args = parser.parse_args()
    output = args.research_output
    output.mkdir(parents=True, exist_ok=False)
    server_args = ServerArgs.from_cli_args(args)
    expected_revision = 'b968826d9c46dd6066d109eabc6255188de91218'
    assert server_args.model_path == 'Qwen/Qwen3-8B' and server_args.revision == expected_revision
    assert server_args.tp_size == 8 and len(jax.devices()) == 8
    save(output / 'requested-server-args.json', asdict(server_args))
    save(output / 'packages.json', sorted((d.metadata['Name'], d.version) for d in metadata.distributions()))
    snapshot = Path(snapshot_download(server_args.model_path, revision=expected_revision,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja']))
    weights = []
    for p in sorted(snapshot.iterdir()):
        if not p.is_file():
            continue
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(8 << 20), b''):
                h.update(chunk)
        weights.append({'name': p.name, 'sha256': h.hexdigest(), 'size_bytes': p.stat().st_size})
    assert any(p['name'].endswith('.safetensors') for p in weights)
    save(output / 'checkpoint.json', {'revision': expected_revision, 'files': weights})
    server_args.model_path = str(snapshot)
    server_args.tokenizer_path = str(snapshot)
    server_args.cuda_graph_max_bs = 8
    _set_envs_and_config(server_args)
    # The pinned bench.load_model omits the now-required dp_size argument.
    # Use the same constructors with the explicit CLI value; no source patch.
    mesh = bench.create_device_mesh(ici_parallelism=[1, 8], dcn_parallelism=[1, 1])
    class BenchModelRunner(bench.ModelRunner):
        def forward(self, *args, **kwargs):
            logits, cache_misses, layers_topk_ids = super().forward(*args, **kwargs)
            # Current TpModelWorker consumes the third MoE routing result, while
            # this older benchmark expects two values. Dense Qwen has no routes.
            assert layers_topk_ids is None
            return logits, cache_misses

    model = BenchModelRunner(model_config=bench.ModelConfig.from_server_args(server_args),
        mem_fraction_static=server_args.mem_fraction_static, tp_size=8,
        dp_size=server_args.dp_size, server_args=server_args, mesh=mesh)
    # TpModelWorker normally initializes this buffer after choosing buckets;
    # the standalone benchmark bypasses that worker. Cover the fixed KV cap.
    model.req_to_token_pool.init_cache_loc_host_buffer(server_args.max_total_tokens)
    tokenizer = bench.get_tokenizer(str(snapshot), tokenizer_mode=server_args.tokenizer_mode,
                                   trust_remote_code=False)
    from sgl_jax.srt.models import qwen3
    model_file = Path(inspect.getfile(qwen3))
    save(output / 'model-source.json', {'module': str(model_file),
        'sha256': hashlib.sha256(model_file.read_bytes()).hexdigest(), 'modified': False})
    save(output / 'resolved-server-args.json', asdict(server_args))
    rng = np.random.default_rng(20260915)
    text_prompts = [
        '请用一句话概括：团队完成了编译器测试，随后验证了模型的精度和显存使用情况。摘要：',
        'Question: Why does water freeze at low temperatures? Answer:',
        'Calculate 17 + 25. The answer is',
        'Complete the Python function:\ndef square(x):\n    return',
    ]
    cases = [(f'tokens-{n}', rng.integers(100, 10000, size=(1, n)).tolist()) for n in (512, 4096, 8192)]
    cases.append(('text-regression', [tokenizer.encode(p) for p in text_prompts]))
    save(output / 'text-prompts.json', text_prompts)
    results = []
    for name, inputs in cases:
        case = output / name
        case.mkdir()
        save(case / 'input-ids.json', inputs)
        previous = None
        repetitions = []
        # First invocation includes cold compilation. A separate profile run is
        # excluded from the unprofiled timings; both save actual token outputs.
        for rep in range(3):
            model.req_to_token_pool.clear()
            model.token_to_kv_pool_allocator.clear()
            reqs = bench.prepare_synthetic_inputs_for_latency_test(len(inputs), len(inputs[0]), inputs)
            for req in reqs:
                req.sampling_params.max_new_tokens = 32
            profiler = jax.profiler.trace(str(case / 'profile')) if rep == 2 else nullcontext()
            timings = []
            tokens = [[] for _ in inputs]
            with profiler:
                for step in range(32):
                    start = time.perf_counter()
                    with jax.profiler.StepTraceAnnotation('qwen3_baseline', step_num=step):
                        if step == 0:
                            ids, logits, batch = bench.extend(reqs, model)
                        else:
                            ids, logits = bench.decode(ids_cpu, batch, model)
                        ids_cpu = np.asarray(ids.block_until_ready())
                    timings.append(time.perf_counter() - start)
                    logits_cpu = np.asarray(logits.block_until_ready()).astype(np.float32)
                    assert np.all(np.isfinite(logits_cpu))
                    if step == 0:
                        np.save(case / f'prefill-logits-{rep}.npy', logits_cpu, allow_pickle=False)
                    for i in range(len(inputs)):
                        tokens[i].append(int(ids_cpu[i]))
            save(case / f'output-ids-{rep}.json', tokens)
            if previous is not None:
                assert tokens == previous, f'greedy repeatability failed: {name}, {rep}'
            previous = tokens
            repetitions.append({'rep': rep, 'cold': rep == 0, 'profiled': rep == 2,
                'step_seconds': timings, 'device_memory_stats_after': [d.memory_stats() for d in jax.devices()]})
        save(case / 'runs.json', repetitions)
        save(case / 'generated-text.json', [tokenizer.decode(t) for t in previous])
        results.append({'case': name, 'input_lengths': list(map(len, inputs)), 'repeatable': True})
        save(output / 'progress.json', results)
        print('QWEN3_CASE_OK', name, flush=True)
    save(output / 'result.json', {'evidence_level': 'RUN-TPU', 'qualifiers': ['VERSION-SKEW'],
         'real_checkpoint': True, 'cases': results, 'compiler_modified': False,
         'fusion_split_accepted': False, 'timing_scope': 'one cold, one warm, one profiled run; not a performance claim'})
    print('QWEN3_REAL_BASELINE_OK', flush=True)


if __name__ == '__main__':
    main()
