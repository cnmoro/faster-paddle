"""Bounded CPU graph/traffic probe; requires numpy, onnx and onnxruntime.

Run under taskset/nice when sharing a machine. Defaults to one inference thread,
sequential sessions, and five timed runs. JSONL goes to stdout. This estimates
logical tensor traffic, NOT DRAM traffic or hardware bandwidth utilization.
Model variants exist only in memory; bundled model files are never modified.
"""
import argparse
from collections import defaultdict
import copy
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


def options(threads, profile=None):
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = threads
    opt.inter_op_num_threads = 1
    opt.enable_mem_pattern = False
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.add_session_config_entry('session.force_spinning_stop', '1')
    if profile:
        opt.enable_profiling = True
        opt.profile_file_prefix = profile
    return opt


def specialize(model, shape):
    model = copy.deepcopy(model)
    for dim, value in zip(model.graph.input[0].type.tensor_type.shape.dim, shape):
        dim.ClearField('dim_param')
        dim.dim_value = value
    return model


def compact_output(model):
    model = copy.deepcopy(model)
    output = model.graph.output[0].name
    # Keep Softmax and its probabilities: preserve the confidence contract.
    model.graph.node.extend([
        helper.make_node('ArgMax', [output], ['probe_ids'], axis=2, keepdims=0,
                         select_last_index=0),
        helper.make_node('ReduceMax', [output], ['probe_scores'], axes=[2], keepdims=0),
    ])
    del model.graph.output[:]
    model.graph.output.extend([
        helper.make_tensor_value_info('probe_ids', TensorProto.INT64, [None, None]),
        helper.make_tensor_value_info('probe_scores', TensorProto.FLOAT, [None, None]),
    ])
    onnx.checker.check_model(model)
    return model


def graph_stats(model, shape):
    inferred = onnx.shape_inference.infer_shapes(specialize(model, shape), data_prop=True)
    shapes, sizes = {}, {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        typ = value.type.tensor_type
        if typ.HasField('shape') and all(d.HasField('dim_value') for d in typ.shape.dim):
            dims = [d.dim_value for d in typ.shape.dim]
            shapes[value.name] = dims
            sizes[value.name] = math.prod(dims) * np.dtype(helper.tensor_dtype_to_np_dtype(typ.elem_type)).itemsize
    weights = 0
    for tensor in inferred.graph.initializer:
        shapes[tensor.name] = list(tensor.dims)
        size = math.prod(tensor.dims) * np.dtype(helper.tensor_dtype_to_np_dtype(tensor.data_type)).itemsize
        sizes[tensor.name] = size
        weights += size
    flops, traffic = defaultdict(int), defaultdict(int)
    unresolved = defaultdict(int)
    for node in inferred.graph.node:
        edges = [name for name in list(node.input) + list(node.output) if name]
        if all(name in sizes for name in edges):
            traffic[node.op_type] += sum(sizes[name] for name in edges)
        else:
            unresolved[node.op_type] += 1
        if node.op_type == 'Conv' and node.output[0] in shapes and node.input[1] in shapes:
            flops['Conv'] += 2 * math.prod(shapes[node.output[0]]) * math.prod(shapes[node.input[1]][1:])
        if node.op_type == 'MatMul' and node.output[0] in shapes and node.input[0] in shapes:
            flops['MatMul'] += 2 * math.prod(shapes[node.output[0]]) * shapes[node.input[0]][-1]
    return dict(weight_bytes=weights, conv_matmul_flops=dict(flops),
                logical_edge_bytes_by_op=dict(traffic), unknown_traffic_nodes=dict(unresolved),
                accounting='Original graph edges, including identities; not optimized physical traffic. FLOPs count only resolved Conv/MatMul.')


def run_case(size, stage, width, repeat, threads, variants, output_dir):
    path = Path(__file__).resolve().parents[1] / 'models' / size / (stage + '.onnx')
    model = onnx.load(path)
    shape = [1, 3, 48, width] if stage == 'rec' else [1, 3, 736, 1184]
    x = np.random.default_rng(7).uniform(-1, 1, shape).astype(np.float32)
    base = dict(model=size, stage=stage, input_shape=shape, threads=threads, runtime=ort.__version__,
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)
    print(json.dumps(dict(base, kind='graph', **graph_stats(model, shape))), flush=True)
    ref = None
    names = [name for name in variants if stage == 'rec' or 'compact' not in name]
    sessions = {}
    # Alternate variant order between rounds; each session is run sequentially.
    for name in names:
        variant = specialize(model, shape) if name.startswith('static') else model
        if 'compact' in name:
            variant = compact_output(variant)
        t = time.perf_counter()
        session = ort.InferenceSession(variant.SerializeToString(), options(threads), providers=['CPUExecutionProvider'])
        init_ms = (time.perf_counter() - t) * 1000
        output = session.run(None, {'x': x})
        if name == 'dynamic':
            ref = output[0]
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                np.save(output_dir / f'{size}-{stage}-{shape[-1]}.npy', ref)
        if 'compact' in name:
            matches = bool(np.array_equal(output[0], ref.argmax(axis=2)))
            delta = float(np.max(np.abs(output[1] - ref.max(axis=2))))
        else:
            matches = bool(np.array_equal(output[0].argmax(axis=-1), ref.argmax(axis=-1)))
            delta = float(np.max(np.abs(output[0] - ref)))
        sessions[name] = (session, [], dict(init_ms=init_ms, output_bytes=sum(y.nbytes for y in output),
                                           argmax_matches=matches, max_abs_difference=delta))
    for i in range(repeat):
        for name in names[::1 if i % 2 == 0 else -1]:
            session, samples, _ = sessions[name]
            t = time.perf_counter()
            session.run(None, {'x': x})
            samples.append((time.perf_counter() - t) * 1000)
    for name in names:
        _, samples, details = sessions[name]
        print(json.dumps(dict(base, kind='timing', variant=name, samples_ms=samples,
                              median_ms=statistics.median(samples), **details)), flush=True)
    sessions.clear()
    del session
    with tempfile.TemporaryDirectory(prefix='ocr-memory-profile-') as tmp:
        session = ort.InferenceSession(model.SerializeToString(), options(threads, str(Path(tmp)/'trace')),
                                       providers=['CPUExecutionProvider'])
        session.run(None, {'x': x})
        session.run(None, {'x': x})
        with open(session.end_profiling()) as f:
            events = json.load(f)
        times = defaultdict(float)
        # Exclude cold run. ORT emits model_run after that run's kernel events.
        first_done = False
        for event in events:
            if event.get('name') == 'model_run':
                first_done = True
            elif first_done and event.get('cat') == 'Node' and event['name'].endswith('_kernel_time'):
                times[event['args'].get('op_name', '?')] += event['dur']
        print(json.dumps(dict(base, kind='profile', warm_kernel_us=dict(sorted(times.items(), key=lambda kv: -kv[1])))), flush=True)
        del session


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--models', nargs='+', choices=['tiny', 'small'], default=['tiny', 'small'])
    parser.add_argument('--detector', action='store_true')
    parser.add_argument('--output-dir', type=Path, help='Save baseline arrays for cross-runtime numerical comparison')
    parser.add_argument('--variants', nargs='+', choices=['dynamic', 'static', 'compact', 'static_compact'],
                        default=['dynamic', 'static', 'compact', 'static_compact'])
    args = parser.parse_args()
    if args.repeat < 1 or args.threads < 1:
        parser.error('repeat and threads must be positive')
    if args.variants[0] != 'dynamic' or len(set(args.variants)) != len(args.variants):
        parser.error('variants must start with dynamic and contain no duplicates')
    for size in args.models:
        for width in [64 if size == 'tiny' else 96, 320]:
            run_case(size, 'rec', width, args.repeat, args.threads, args.variants, args.output_dir)
        if args.detector:
            run_case(size, 'det', None, args.repeat, args.threads, args.variants, args.output_dir)
