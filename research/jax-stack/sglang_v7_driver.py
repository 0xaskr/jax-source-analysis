#!/usr/bin/env python3
"""Drive real Qwen3 inference through the native SGLang prefill/decode path.

Why this exists instead of reusing `sgl_jax.bench_one_batch`:

  * Its shared helper `_run_forward_and_sample` unpacks two values from
    `ModelRunner.forward`, but `ModelRunner._forward` returns three
    `(output, cache_miss_count, layers_topk_ids)`. The helper cannot run.
  * Its `latency_test` path never maintains `output_ids`, and the revision
    carries an explicit `# TODO: Fix this function` on it.
  * Running the standalone benchmark also bypasses `TpModelWorker.__init__`,
    which is what sizes the persistent `cache_loc` host buffer.

This driver therefore calls the same public building blocks the benchmark
builds on (`ScheduleBatch.init_new`, `prepare_for_extend`,
`prepare_for_decode`, `ForwardBatch.init_new`, `LogitsMetadata`,
`SamplingMetadata`) and does the three-value unpacking itself. No business
model file and no installed SGLang source file is modified.

Deliberate, recorded differences from the pinned benchmark:
  * `cache_loc` host buffer sized from `CompilationManager`, like
    `TpModelWorker.__init__`, not from an arbitrary token cap.
  * generated token ids gathered through
    `jax.experimental.multihost_utils.process_allgather` before use, as every
    native benchmark call site does.
  * greedy repeatability asserted on exported arrays, not loop-local state.

`--preflight-only` stops before the checkpoint download and requires no TPU; it
validates imports, CLI parsing and device assumptions only.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
from pathlib import Path
import time
from types import SimpleNamespace

MODEL = 'Qwen/Qwen3-8B'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
SEED = 20260915
TP = 8
DP = 1
MAX_NEW_TOKENS = 32
LENGTHS = (512, 4096, 8192)
REPETITIONS = 3          # rep 0 cold, rep 1 warm, rep 2 profiled
PROFILE_REP = 2
TEXT_PROMPTS = [
    '请用一句话概括：团队完成了编译器测试，随后验证了模型的精度和显存使用情况。摘要：',
    'Question: Why does water freeze at low temperatures? Answer:',
    'Calculate 17 + 25. The answer is',
    'Complete the Python function:\ndef square(x):\n    return',
]
SERVER_CLI = [
    '--model-path', MODEL, '--revision', REVISION,
    '--tp-size', str(TP), '--dp-size', str(DP), '--device', 'tpu', '--dtype', 'bfloat16',
    '--attention-backend', 'fa', '--page-size', '128', '--max-running-requests', '8',
    '--max-total-tokens', '65536', '--max-prefill-tokens', '8192',
    '--chunked-prefill-size', '8192', '--context-length', '32768',
    '--mem-fraction-static', '0.4', '--disable-radix-cache', '--disable-precompile',
    '--random-seed', str(SEED),
]


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true',
                        help='validate imports/CLI/devices and stop before the checkpoint')
    args = parser.parse_args()
    output = args.output
    preflight_only = args.preflight_only

    import jax
    import numpy as np
    from jax.experimental import multihost_utils as jax_mh
    from sgl_jax import bench_one_batch as bench
    from sgl_jax.srt.entrypoints.engine import _set_envs_and_config
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
    from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
    from sgl_jax.srt.model_executor.model_runner import ModelRunner
    from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
    from sgl_jax.srt.server_args import PortArgs, ServerArgs
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    # Parse the server stack into its own parser/namespace: reusing the driver
    # parser here would overwrite the already-parsed --output value.
    server_parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(server_parser)
    server_args = ServerArgs.from_cli_args(server_parser.parse_args(SERVER_CLI))
    assert server_args.model_path == MODEL and server_args.revision == REVISION
    assert server_args.tp_size == TP and server_args.dp_size == DP
    assert server_args.page_size == 128 and server_args.context_length == 32768
    assert server_args.disable_radix_cache and server_args.disable_precompile

    # Pin the API shape this driver depends on, so silent upstream drift is loud.
    forward_src = inspect.getsource(ModelRunner._forward)
    assert 'return output, cache_miss_count, layers_topk_ids' in forward_src, (
        'ModelRunner._forward no longer returns three values; recheck the driver')
    assert 'logits_output, _ = model_runner.forward' in inspect.getsource(
        bench._run_forward_and_sample), (
        'the pinned benchmark helper changed; re-evaluate whether it is usable again')
    assert hasattr(CompilationManager, '_compute_cache_loc_buckets'), (
        'CompilationManager no longer computes cache_loc buckets; resize the host buffer check')
    backend_defines_limit = []
    for module_name in ('sgl_jax.srt.layers.attention.flashattention_backend',
                        'sgl_jax.srt.layers.attention.native_backend',
                        'sgl_jax.srt.layers.attention.mla_backend'):
        module = __import__(module_name, fromlist=['*'])
        if any(inspect.isclass(attr) and hasattr(attr, 'get_max_running_reqests')
               for attr in vars(module).values()):
            backend_defines_limit.append(module_name)
    assert backend_defines_limit, (
        'no attention backend exposes get_max_running_reqests; the max_running_requests '
        'clamp this driver mirrors no longer exists')

    # libtpu is absent in a CPU-only checkout on purpose, so record it as null
    # rather than failing; the TPU run asserts it explicitly further down.
    def pkg_version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    packages = {n: pkg_version(n) for n in ('jax', 'jaxlib', 'libtpu')}
    devices = jax.devices()
    output.mkdir(parents=True, exist_ok=False)
    preflight = {
        'python': platform.python_version(),
        'packages': packages,
        'device_count': len(devices),
        # coords/core_on_chip exist only on TPU device objects, so a CPU preflight
        # must not fail on them.
        'devices': [{'id': d.id, 'kind': d.device_kind, 'platform': d.platform,
                     'coords': list(d.coords) if hasattr(d, 'coords') else None,
                     'core_on_chip': getattr(d, 'core_on_chip', None)}
                    for d in devices],
        'cli_parsed_server_args': asdict(server_args),
        'api_shape_pins': {
            'ModelRunner._forward_returns_three': True,
            'bench_helper_unpacks_two': True,
            'business_source_modified': False,
        },
    }
    save(output / 'preflight.json', preflight)
    if preflight_only:
        print('DRIVER_PREFLIGHT_ONLY_OK', json.dumps(packages), flush=True)
        return

    assert len(devices) == TP, f'expected exactly {TP} devices, got {len(devices)}'
    assert all(d.platform == 'tpu' for d in devices), 'not all devices are TPU'
    assert packages['libtpu'] is not None, 'libtpu metadata missing on a TPU run'
    save(output / 'identity.json', {
        **preflight,
        'model': MODEL, 'model_revision': REVISION, 'seed': SEED,
        'evidence_level': 'RUN-TPU', 'qualifiers': ['VERSION-SKEW'],
        'source_alignment': 'SGLang compatibility stack; not the pinned research JAX/XLA',
        'entry_point': 'ModelRunner + ScheduleBatch native prefill/decode',
        'generated_token_ids_scope': 'baseline; fusion/split candidates must match these',
    })
    save(output / 'packages.json',
         sorted((d.metadata['Name'], d.version) for d in metadata.distributions()))

    # ------------------------------------------------------------ checkpoint
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(
        MODEL, revision=REVISION,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja']))
    files = [{'name': p.name, 'sha256': sha256_file(p), 'size_bytes': p.stat().st_size}
             for p in sorted(snapshot.iterdir()) if p.is_file()]
    assert any(f['name'].endswith('.safetensors') for f in files), 'no safetensors downloaded'
    save(output / 'checkpoint.json',
         {'revision': REVISION, 'snapshot': str(snapshot), 'files': files})
    server_args.model_path = str(snapshot)
    server_args.tokenizer_path = str(snapshot)
    _set_envs_and_config(server_args)

    # ---------------------------------------------------------------- model
    mesh = create_device_mesh(ici_parallelism=[1, TP], dcn_parallelism=[1, 1])
    port_args = PortArgs.init_new(server_args)
    model_runner = ModelRunner(
        model_config=bench.ModelConfig.from_server_args(server_args),
        mem_fraction_static=server_args.mem_fraction_static,
        tp_size=TP, dp_size=DP, server_args=server_args, mesh=mesh)

    # Size the persistent cache_loc host buffer the way TpModelWorker does.
    # Mirrors TpModelWorker.__init__ + get_max_padded_size: max_running_requests
    # is itself clamped by server, pool and attention-backend limits, so the
    # server flag alone is not the value the worker actually uses.
    attn_backend_limit = (
        model_runner.attn_backend.get_max_running_reqests(
            model_runner.model_config.context_len, server_args.page_size) * DP)
    server_limit = (model_runner.max_total_num_tokens // 2
                    if server_args.max_running_requests is None
                    else server_args.max_running_requests)
    pool_limit = model_runner.req_to_token_pool.size
    effective_max_running_requests = min(server_limit, pool_limit, attn_backend_limit)
    if effective_max_running_requests % DP:
        effective_max_running_requests = (effective_max_running_requests // DP) * DP
    max_padded_batch_size = min(server_args.chunked_prefill_size
                                if server_args.chunked_prefill_size > 0
                                else server_args.max_prefill_tokens,
                                effective_max_running_requests)
    max_padded_num_tokens = (server_args.chunked_prefill_size
                             if server_args.chunked_prefill_size > 0
                             else server_args.max_prefill_tokens)
    max_req_len = min(server_args.context_length - 1, model_runner.max_total_num_tokens - 1)
    manager = CompilationManager(
        server_args=server_args, max_padded_batch_size=max_padded_batch_size,
        max_padded_num_tokens=max_padded_num_tokens, dp_size=DP, tp_size=TP,
        page_size=server_args.page_size, max_req_len=max_req_len,
        vocab_size=model_runner.model_config.vocab_size,
        multimodal=server_args.multimodal,
        has_recurrent_state=model_runner.linear_recurrent_config is not None)
    cache_loc_cap = manager.cache_loc_buckets[-1]
    model_runner.req_to_token_pool.init_cache_loc_host_buffer(cache_loc_cap)
    save(output / 'cache-loc-sizing.json', {
        'max_total_num_tokens': model_runner.max_total_num_tokens,
        'max_req_len': max_req_len,
        'requested_max_running_requests': server_args.max_running_requests,
        'constraint_server_limit': server_limit,
        'constraint_pool_limit': pool_limit,
        'constraint_attn_backend_limit': attn_backend_limit,
        'effective_max_running_requests': effective_max_running_requests,
        'max_padded_batch_size': max_padded_batch_size,
        'max_padded_num_tokens': max_padded_num_tokens,
        'token_buckets': manager.token_buckets, 'bs_buckets': manager.bs_buckets,
        'cache_loc_buckets': manager.cache_loc_buckets, 'cache_loc_cap_used': cache_loc_cap,
        'note': 'derived from CompilationManager, mirroring TpModelWorker.__init__'})

    tokenizer = bench.get_tokenizer(str(snapshot), tokenizer_mode=server_args.tokenizer_mode,
                                    trust_remote_code=False)
    from sgl_jax.srt.models import qwen3
    model_file = Path(inspect.getfile(qwen3))
    save(output / 'model-source.json', {
        'module': str(model_file), 'sha256': sha256_file(model_file), 'modified': False,
        'mlp_class': 'Qwen3MLP',
        'linear_base': 'sgl_jax.srt.layers.linear.LinearBase'})

    # ------------------------------------------------------------- forward
    def build_batch(reqs, extend):
        batch = ScheduleBatch.init_new(
            reqs=[reqs], req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            tree_cache=SimpleNamespace(
                token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator),
            model_config=model_runner.model_config, enable_overlap=False, dp_size=DP,
            enable_custom_logit_processor=False, chunked_reqs=None)
        if extend:
            batch.prepare_for_extend()
            lens = batch.extend_lens if getattr(batch, 'extend_lens', None) is not None \
                else batch.seq_lens
        else:
            batch.prepare_for_decode()
            lens = batch.seq_lens
        return batch, int(np.sum(np.array(lens, dtype=np.int64)))

    def run_once(batch, token_first_arg):
        """Native forward + sample with the correct three-value unpacking."""
        page_size = model_runner.page_size
        bs_needed = len(batch.seq_lens)
        cache_loc_needed = int(np.sum(
            ((np.array(batch.seq_lens, dtype=np.int64) + page_size - 1) // page_size) * page_size))
        worker_batch = batch.get_model_worker_batch(
            [token_first_arg], [bs_needed], [cache_loc_needed], page_size, False)
        model_runner.attn_backend.forward_metadata = \
            model_runner.attn_backend.get_forward_metadata(worker_batch)
        forward_batch = ForwardBatch.init_new(worker_batch, model_runner)
        logits_metadata = LogitsMetadata.from_model_worker_batch(
            worker_batch, mesh=model_runner.mesh)
        # Three values here; the pinned benchmark helper unpacks two.
        logits_output, cache_miss_count, layers_topk_ids = model_runner.forward(
            forward_batch, logits_metadata=logits_metadata)
        assert layers_topk_ids is None, 'dense Qwen3 must not return MoE routing'
        pad_size = len(worker_batch.seq_lens) - worker_batch.real_bs
        sampling_metadata = SamplingMetadata.from_model_worker_batch(
            worker_batch, pad_size=pad_size, mesh=model_runner.mesh,
            vocab_size=model_runner.model_config.vocab_size)
        next_ids, _, _ = model_runner.sample(logits_output, sampling_metadata)
        return next_ids, logits_output.next_token_logits, cache_miss_count

    # ------------------------------------------------------------------ cases
    rng = np.random.default_rng(SEED)
    cases = [(f'tokens-{n}', [list(map(int, row)) for row in
                              rng.integers(100, 10000, size=(1, n))]) for n in LENGTHS]
    cases.append(('text-regression', [tokenizer.encode(p) for p in TEXT_PROMPTS]))
    save(output / 'text-prompts.json', TEXT_PROMPTS)

    summaries = []
    for name, inputs in cases:
        case = output / name
        case.mkdir()
        save(case / 'input-ids.json', inputs)
        save(case / 'input-lengths.json', [len(i) for i in inputs])
        exported, runs = None, []
        for rep in range(REPETITIONS):
            model_runner.req_to_token_pool.clear()
            model_runner.token_to_kv_pool_allocator.clear()
            reqs = bench.prepare_synthetic_inputs_for_latency_test(
                len(inputs), len(inputs[0]), inputs)
            for r in reqs:
                r.sampling_params.max_new_tokens = MAX_NEW_TOKENS
            profiler = (jax.profiler.trace(str(case / 'profile'), create_perfetto_trace=True)
                        if rep == PROFILE_REP else nullcontext())
            step_seconds, cache_misses, tokens = [], [], [[] for _ in inputs]
            with profiler:
                with jax.profiler.StepTraceAnnotation('qwen3_prefill', step_num=0):
                    start = time.perf_counter()
                    batch, token_first_arg = build_batch(reqs, extend=True)
                    ids, logits, miss = run_once(batch, token_first_arg)
                    ids_cpu = np.asarray(
                        jax_mh.process_allgather(ids, tiled=True).block_until_ready())
                    logits_cpu = np.asarray(jax_mh.process_allgather(
                        logits, tiled=True).block_until_ready()).astype(np.float32)
                    step_seconds.append(time.perf_counter() - start)
                    cache_misses.append(int(miss))
                assert np.all(np.isfinite(logits_cpu)), 'non-finite prefill logits'
                np.save(case / f'prefill-logits-{rep}.npy', logits_cpu, allow_pickle=False)
                for i in range(len(inputs)):
                    tokens[i].append(int(ids_cpu[i]))
                # Native convention: output_len - 1 decode steps after one prefill.
                for step in range(MAX_NEW_TOKENS - 1):
                    with jax.profiler.StepTraceAnnotation('qwen3_decode', step_num=step):
                        start = time.perf_counter()
                        batch, token_first_arg = build_batch(reqs, extend=False)
                        ids, logits, miss = run_once(batch, token_first_arg)
                        ids_cpu = np.asarray(
                            jax_mh.process_allgather(ids, tiled=True).block_until_ready())
                        _ = np.asarray(jax_mh.process_allgather(
                            logits, tiled=True).block_until_ready()).astype(np.float32)
                        step_seconds.append(time.perf_counter() - start)
                        cache_misses.append(int(miss))
                    if ids_cpu.shape[0] != len(inputs):
                        raise AssertionError(
                            f'{name}: decode returned {ids_cpu.shape[0]} ids for {len(inputs)} '
                            f'requests at step {step}; the bench path does not filter finished '
                            f'requests, so batch membership changed')
                    for i in range(len(inputs)):
                        tokens[i].append(int(ids_cpu[i]))
            save(case / f'output-ids-{rep}.json', tokens)
            save(case / f'step-seconds-{rep}.json', step_seconds)
            if exported is not None:
                assert tokens == exported, f'greedy repeatability failed: {name} rep {rep}'
            exported = tokens
            runs.append({
                'rep': rep, 'cold': rep == 0, 'profiled': rep == PROFILE_REP,
                'prompt_tokens_per_request': [len(i) for i in inputs],
                'generated_tokens_per_request': [len(t) for t in tokens],
                'cache_miss_counts': cache_misses,
                'prefill_seconds': step_seconds[0],
                'decode_seconds_median': (float(np.median(step_seconds[1:]))
                                          if len(step_seconds) > 1 else None),
                'decode_seconds_all': step_seconds[1:],
                'device_memory_stats_after': [d.memory_stats() for d in jax.devices()],
            })
        save(case / 'runs.json', runs)
        save(case / 'generated-text.json', [tokenizer.decode(t) for t in exported])
        summaries.append({'case': name, 'input_lengths': [len(i) for i in inputs],
                          'repeatable_across_3_runs': True})
        save(output / 'progress.json', summaries)
        print('QWEN3_CASE_OK', name, flush=True)

    save(output / 'result.json', {
        'evidence_level': 'RUN-TPU', 'qualifiers': ['VERSION-SKEW'],
        'real_checkpoint': True, 'cases': summaries, 'compiler_modified': False,
        'business_source_modified': False,
        'entry_point': 'ModelRunner + ScheduleBatch native prefill/decode; '
                       'not sgl_jax.bench_one_batch._run_forward_and_sample',
        'fusion_split_accepted': False,
        'timing_scope': 'one cold, one warm, one profiled run per case; not a performance claim',
        'greedy_baseline_role': 'exported ids are the reference fusion/split must reproduce',
    })
    print('QWEN3_DRIVER_OK', flush=True)


if __name__ == '__main__':
    main()
