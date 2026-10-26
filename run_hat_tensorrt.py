"""Independent fixed-shape HAT TensorRT experiments; not a production inference path."""
import argparse
import csv
import json
from pathlib import Path
import time
from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from display_model_adapter import (ROOT, ERROR_LIMITS, TEMPORAL_ERROR_LIMITS, infer, load_model,
                                   require_runtime_qualification, within_error_limits)
from display_support import checked_read, prepare, resize, save_image
from ir_sr import read_gray
from runtime_v5 import atomic_json, run_lock, sha, utc_now


INPUT_SHAPES = {2: (256, 320), 3: (171, 213)}


def cached_mask(self, x_size):
    return self._trt_fixed_mask


def fixed_hat_forward(self, x):
    x = (x - self._trt_mean) * self.img_range
    x = self.conv_first(x)
    x = self.conv_after_body(self.forward_features(x)) + x
    x = self.conv_before_upsample(x)
    x = self.conv_last(self.upsample(x))
    return x / self.img_range + self._trt_mean


class FixedHAT(nn.Module):
    def __init__(self, model, shape):
        super().__init__()
        self.model = model
        self.height, self.width = shape
        self.precision = model._display_precision
        padded_shape = tuple((n + 15) // 16 * 16 for n in shape)
        model.register_buffer('_trt_fixed_mask', model.calculate_mask(padded_shape).to('cuda'))
        model.calculate_mask = MethodType(cached_mask, model)
        model.register_buffer('_trt_mean', model.mean.to('cuda').clone())
        model.forward = MethodType(fixed_hat_forward, model)

    def forward(self, gray):
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.precision == 'float16_safe'):
            rgb = gray.repeat(1, 3, 1, 1)
            rgb = F.pad(rgb, (0, (-self.width) % 16, 0, (-self.height) % 16), 'reflect')
            return self.model(rgb)[..., :self.height * 4, :self.width * 4]


def hat_spec(run):
    cfg = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    spec = next(s for s in cfg['models'] if s['name'] == 'hat_real_x4')
    return {**spec, 'runtime_precision': 'float32', 'attention_backend': 'official'}


def raw_to_gray(raw):
    if raw.ndim != 4 or raw.shape[:2] != (1, 3) or not np.isfinite(raw).all():
        raise ValueError('Invalid raw RGB output')
    rgb = np.clip(raw[0].transpose(1, 2, 0), 0, 1)
    return (rgb * np.array([.299, .587, .114], np.float32)).sum(axis=2)


def environment():
    return {'torch_version': torch.__version__, 'torch_cuda_version': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_capability': list(torch.cuda.get_device_capability(0)),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'tf32_enabled': False}


def export_graph(a):
    import onnx
    if a.out.exists():
        raise FileExistsError(a.out)
    spec = {**hat_spec(a.run), 'runtime_precision': a.precision}
    source = read_gray(a.input)
    low, _, geometry = prepare(source, a.scale, roi=a.roi)
    if low.shape != INPUT_SHAPES[a.scale]:
        raise ValueError('This experiment requires the fixed 320x256 or 213x171 ROI')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'exporter_source.py').write_bytes(Path(__file__).read_bytes())
    np.save(a.out / 'prepared_input.npy', low, allow_pickle=False)
    metadata = {'status': 'exporting', 'display_scale': a.scale, 'native_scale': 4,
                'input_shape': [1, 1, *low.shape],
                'output_shape': [1, 3, low.shape[0] * 4, low.shape[1] * 4],
                'run': str(a.run.resolve()), 'protocol_sha256': sha(a.run / 'protocol.json'),
                'source': str(a.input.resolve()), 'source_sha256': sha(a.input),
                'geometry': geometry, 'exporter_sha256': sha(a.out / 'exporter_source.py'),
                'adapter_sha256': sha(ROOT / 'display_model_adapter.py'),
                'support_sha256': sha(ROOT / 'display_support.py'),
                'reader_sha256': sha(ROOT / 'ir_sr.py'),
                'prepared_input_sha256': sha(a.out / 'prepared_input.npy'),
                'onnx_numeric_checked': False,
                'precision': a.precision, 'attention_backend': 'official', 'tf32_enabled': False,
                'graph_changes': ['cache official fixed-shape shifted-window mask',
                                  'register fixed FP32 mean; remove forward-time attribute mutation'],
                'exporter_snapshot': 'exporter_source.py',
                'raw_output': 'unclipped native NCHW RGB; original CPU clip/luminance/PIL resize retained',
                'created_at': utc_now()}
    atomic_json(a.out / 'export.json', metadata)
    try:
        with run_lock(a.run / 'gpu_inference.lock'):
            fp32_reference = None
            if a.precision == 'float16_safe':
                official = hat_spec(a.run)
                reference_model, reference_meta = load_model(official)
                fp32_reference, _, _ = infer(reference_model, official, low)
                metadata['fp32_reference_metadata'] = reference_meta
                metadata['graph_changes'].append('explicit FP16 autocast casts with existing FP32 sensitive-module protection')
                del reference_model
                torch.cuda.empty_cache()
            model, model_meta = load_model(spec)
            metadata['model'] = model_meta
            metadata['environment'] = environment()
            x = torch.from_numpy(low.copy())[None, None].to('cuda')
            with torch.inference_mode():
                ref_gray, _, _ = infer(model, spec, low)
                with torch.autocast('cuda', dtype=torch.float16, enabled=a.precision == 'float16_safe'):
                    original = model(F.pad(x.repeat(1, 3, 1, 1),
                        (0, (-low.shape[1]) % 16, 0, (-low.shape[0]) % 16), 'reflect'))
                original = original[..., :low.shape[0] * 4, :low.shape[1] * 4].float().cpu().numpy()
                wrapper = FixedHAT(model, low.shape).eval()
                candidate = wrapper(x).float().cpu().numpy()
            if not np.isfinite(candidate).all():
                raise ValueError('Fixed-shape wrapper produced nonfinite values')
            max_error = float(np.abs(candidate - original).max())
            gray_error = float(np.abs(raw_to_gray(candidate) - ref_gray).max())
            metadata['wrapper_check'] = {'raw_max_abs': max_error, 'native_gray_max_abs': gray_error}
            metadata['wrapper_check']['reference_precision'] = a.precision
            if max_error > 1e-6 or gray_error > 1e-6:
                raise ValueError('Fixed-shape wrapper differs from the selected PyTorch runtime before export')
            if fp32_reference is not None:
                native = raw_to_gray(candidate)
                probe = {'native_error': error_metrics(native, fp32_reference),
                         'display_error': error_metrics(resize(native, (640, 512)), resize(fp32_reference, (640, 512)))}
                metadata['fp32_probe'] = probe
                if not all(within_error_limits(e) for e in probe.values()):
                    raise ValueError('Mixed-precision export probe exceeds unchanged FP32 error limits')
            tick = time.perf_counter()
            torch.onnx.export(wrapper, (x,), a.out / 'hat.onnx',
                input_names=['gray'], output_names=['raw_rgb'],
                dynamo=True, opset_version=20, optimize=True, external_data=True,
                report=True, artifacts_dir=str(a.out))
            metadata['export_seconds'] = time.perf_counter() - tick
            onnx.checker.check_model(str(a.out / 'hat.onnx'), full_check=True)
            graph = onnx.load(str(a.out / 'hat.onnx'), load_external_data=False)
            metadata['node_counts'] = {op: sum(n.op_type == op for n in graph.graph.node)
                                       for op in sorted({n.op_type for n in graph.graph.node})}
            metadata['initializer_types'] = {onnx.TensorProto.DataType.Name(dtype):
                sum(t.data_type == dtype for t in graph.graph.initializer)
                for dtype in sorted({t.data_type for t in graph.graph.initializer})}
            files = [a.out / 'hat.onnx', a.out / 'prepared_input.npy', *sorted(a.out.glob('*.data'))]
            metadata['artifacts'] = [{'file': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size}
                                     for p in files]
            metadata['onnx_version'] = onnx.__version__
            metadata['status'] = 'exported'
            atomic_json(a.out / 'export.json', metadata)
            print(json.dumps({'status': 'exported', 'out': str(a.out),
                              'wrapper_check': metadata['wrapper_check']}, ensure_ascii=False), flush=True)
    except Exception as error:
        metadata.update(status='failed', error=str(error), updated_at=utc_now())
        atomic_json(a.out / 'export.json', metadata)
        raise


def validate_graph(folder, run):
    meta = json.loads((folder / 'export.json').read_text(encoding='utf-8'))
    if meta['status'] != 'exported' or meta['precision'] not in ('float32', 'float16_safe'):
        raise ValueError('Expected completed FP32 or protected mixed-precision export')
    if meta['protocol_sha256'] != sha(run / 'protocol.json'):
        raise ValueError('Export belongs to a different protocol')
    for key, path in (('adapter_sha256', ROOT / 'display_model_adapter.py'),
                      ('support_sha256', ROOT / 'display_support.py'),
                      ('reader_sha256', ROOT / 'ir_sr.py')):
        if meta.get(key) != sha(path):
            raise ValueError('Reference adapter or preprocessing changed since export')
    if sha(folder / 'prepared_input.npy') != meta['prepared_input_sha256']:
        raise ValueError('Prepared export input changed')
    if sha(folder / meta['exporter_snapshot']) != meta['exporter_sha256']:
        raise ValueError('Exporter snapshot hash mismatch')
    model = meta['model']
    for path, expected in ((model['spec']['weight'], model['weight_sha256']),
                           (model['architecture_file'], model['architecture_sha256']),
                           (model['config_file'], model['config_sha256']),
                           (ROOT / 'third_party/BasicSR_display/basicsr/archs/arch_util.py', model['helper_sha256'])):
        if sha(path) != expected:
            raise ValueError('Weights or upstream implementation changed since export')
    for item in meta['artifacts']:
        if Path(item['file']).name != item['file'] or sha(folder / item['file']) != item['sha256']:
            raise ValueError('ONNX artifact hash mismatch')
    return meta


def build_engine(a):
    import tensorrt as trt
    if a.out.exists():
        raise FileExistsError(a.out)
    exported = validate_graph(a.graph, a.run)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'builder_source.py').write_bytes(Path(__file__).read_bytes())
    metadata = {'status': 'building', 'graph': str(a.graph.resolve()),
                'export_metadata_sha256': sha(a.graph / 'export.json'),
                'builder_sha256': sha(a.out / 'builder_source.py'),
                'tensorrt_version': trt.__version__, 'environment': environment(),
                'precision': exported['precision'], 'tf32_enabled': False,
                'workspace_bytes': a.workspace_gib * 1024**3,
                'optimization_level': a.optimization_level, 'created_at': utc_now()}
    atomic_json(a.out / 'build.json', metadata)
    try:
        with run_lock(a.run / 'gpu_inference.lock'):
            class BuildLogger(trt.ILogger):
                def __init__(self):
                    super().__init__()
                    self.path = a.out / 'tensorrt.log'
                    self.path.write_text('', encoding='utf-8')

                def log(self, severity, message):
                    if severity in (trt.Logger.INTERNAL_ERROR, trt.Logger.ERROR,
                                    trt.Logger.WARNING, trt.Logger.INFO):
                        line = f'[{severity}] {message}\n'
                        with self.path.open('a', encoding='utf-8') as stream:
                            stream.write(line)
                        print(line, end='', flush=True)

            logger = BuildLogger()
            metadata['diagnostic_log'] = logger.path.name
            builder = trt.Builder(logger)
            network = builder.create_network()
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(str(a.graph / 'hat.onnx')):
                raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            if network.num_inputs != 1 or network.num_outputs != 1:
                raise ValueError('Unexpected network I/O')
            if list(network.get_input(0).shape) != exported['input_shape'] or list(network.get_output(0).shape) != exported['output_shape']:
                raise ValueError('Parsed graph shape differs from fixed export')
            if network.get_input(0).dtype != trt.float32 or network.get_output(0).dtype != trt.float32:
                raise ValueError('Expected FP32 graph I/O')
            for tensor in (network.get_input(0), network.get_output(0)):
                tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
            config = builder.create_builder_config()
            config.clear_flag(trt.BuilderFlag.TF32)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, metadata['workspace_bytes'])
            config.builder_optimization_level = a.optimization_level
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            metadata['tf32_enabled'] = config.get_flag(trt.BuilderFlag.TF32)
            tick = time.perf_counter()
            plan = builder.build_serialized_network(network, config)
            metadata['build_seconds'] = time.perf_counter() - tick
            if plan is None:
                raise RuntimeError('TensorRT failed to build an engine; inspect the build log')
            (a.out / 'hat.engine').write_bytes(bytes(plan))
            cache = config.get_timing_cache()
            if cache is not None:
                (a.out / 'timing.cache').write_bytes(bytes(cache.serialize()))
            metadata.update(status='built', engine_sha256=sha(a.out / 'hat.engine'),
                            engine_bytes=(a.out / 'hat.engine').stat().st_size,
                            input_shape=exported['input_shape'], output_shape=exported['output_shape'],
                            diagnostic_log_sha256=sha(a.out / 'tensorrt.log'))
            atomic_json(a.out / 'build.json', metadata)
            print(json.dumps({'status': 'built', 'out': str(a.out),
                              'build_seconds': metadata['build_seconds']}, ensure_ascii=False), flush=True)
    except Exception as error:
        if (a.out / 'tensorrt.log').exists():
            metadata['diagnostic_log_sha256'] = sha(a.out / 'tensorrt.log')
        metadata.update(status='failed', error=str(error), updated_at=utc_now())
        atomic_json(a.out / 'build.json', metadata)
        raise


class EngineRunner:
    def __init__(self, folder, run):
        import tensorrt as trt
        self.meta = json.loads((folder / 'build.json').read_text(encoding='utf-8'))
        meta = self.meta
        if meta['status'] != 'built' or meta['tensorrt_version'] != trt.__version__:
            raise ValueError('Incomplete engine or different TensorRT version')
        if meta['tf32_enabled'] or sha(folder / 'hat.engine') != meta['engine_sha256']:
            raise ValueError('TF32 enabled or engine hash mismatch')
        if sha(folder / 'builder_source.py') != meta['builder_sha256']:
            raise ValueError('Builder source snapshot hash mismatch')
        graph = Path(meta['graph'])
        if sha(graph / 'export.json') != meta['export_metadata_sha256']:
            raise ValueError('Export metadata changed after engine build')
        self.exported = validate_graph(graph, run)
        if meta['environment']['gpu_name'] != torch.cuda.get_device_name(0):
            raise ValueError('Engine was built for a different GPU')
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine((folder / 'hat.engine').read_bytes())
        if self.engine is None:
            raise RuntimeError('Engine deserialization failed')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Execution context creation failed')
        if self.engine.num_io_tensors != 2:
            raise ValueError('Expected one input and one output')
        self.stream = torch.cuda.Stream()
        self.input = torch.empty(meta['input_shape'], dtype=torch.float32, device='cuda')
        self.output = torch.empty(meta['output_shape'], dtype=torch.float32, device='cuda')
        for name, tensor, mode in (('gray', self.input, trt.TensorIOMode.INPUT),
                                   ('raw_rgb', self.output, trt.TensorIOMode.OUTPUT)):
            if self.engine.get_tensor_mode(name) != mode or self.engine.get_tensor_dtype(name) != trt.float32:
                raise ValueError('Engine tensor mode or dtype mismatch')
            if self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR or self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError('Expected contiguous, nonvectorized GPU engine I/O')
            if list(self.engine.get_tensor_shape(name)) != list(tensor.shape):
                raise ValueError('Engine tensor shape mismatch')
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError('Failed to register engine tensor address')
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def infer(self, gray):
        gray = np.asarray(gray, np.float32)
        if gray.shape != tuple(self.meta['input_shape'][2:]) or not np.isfinite(gray).all() or gray.min() < 0 or gray.max() > 1:
            raise ValueError('Expected finite normalized grayscale with the fixed engine shape')
        torch.cuda.synchronize()
        tick = time.perf_counter()
        host = torch.from_numpy(gray.copy())[None, None].pin_memory()
        with torch.cuda.stream(self.stream):
            self.input.copy_(host, non_blocking=True)
            self.start.record(self.stream)
            if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
                raise RuntimeError('TensorRT enqueue failed')
            self.end.record(self.stream)
        self.stream.synchronize()
        raw = self.output.cpu().numpy().copy()
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite TensorRT raw output')
        timing = {'inference_ms': (time.perf_counter() - tick) * 1000,
                  'gpu_forward_ms': self.start.elapsed_time(self.end),
                  'raw_min': float(raw.min()), 'raw_max': float(raw.max())}
        return raw_to_gray(raw), timing


def error_metrics(pred, ref):
    difference = np.abs(pred - ref) * 255
    return {'mean_abs_dn': float(difference.mean()),
            'p99_abs_dn': float(np.percentile(difference, 99)),
            'max_abs_dn': float(difference.max())}


def experiment_cases(run, scale, indices, consecutive, hard):
    cfg = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    requested = set(indices) | set(range(indices[0], indices[0] + consecutive))
    cases = []
    for sequence, group in cfg['groups'].items():
        records = {int(r['frame_index']): r for r in cfg['source_inputs'] if r['sequence'] == sequence}
        if not requested <= records.keys():
            raise ValueError('Requested source frames absent from frozen protocol')
        for index in sorted(requested):
            record = records[index]
            source = checked_read(ROOT / cfg['source_data'], record)
            low, _, geometry = prepare(source, scale, roi=group['roi'][str(scale)])
            cases.append({'sequence': sequence, 'role': group['role'], 'index': index,
                          'scale': scale, 'source': record, 'geometry': geometry,
                          'low': low, 'benchmark': index in indices})
    if hard:
        root = ROOT / 'data/rosbag_frames_hard_v1'
        for table in sorted(root.glob('*/frames.csv')):
            with table.open(encoding='utf-8', newline='') as stream:
                records = [r for r in csv.DictReader(stream) if int(r['frame_index']) in (74, 75, 76)]
            for record in records:
                source = checked_read(root, record)
                low, _, geometry = prepare(source, scale)
                cases.append({'sequence': record['sequence'], 'role': 'hard_development',
                              'index': int(record['frame_index']), 'scale': scale, 'source': record,
                              'geometry': geometry, 'low': low, 'benchmark': False})
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        sequences = {s['name'] for bag in manifest['bags'] for s in bag['sequences']}
        selected = [(c['sequence'], c['index']) for c in cases if c['role'] == 'hard_development']
        expected = {(s, i) for s in sequences for i in (74, 75, 76)}
        if len(sequences) != 4 or len(selected) != 12 or set(selected) != expected:
            raise ValueError('Hard-sample selection requires four distinct sequences and unique frames 74/75/76')
    if not cases or any(c['low'].shape != INPUT_SHAPES[scale] for c in cases):
        raise ValueError('Empty cases or ROI shape differs from the fixed engine')
    return cases


def check_engine(a):
    if a.out.exists():
        raise FileExistsError(a.out)
    if a.repeats < 1 or a.warmup < 1 or a.consecutive < 0:
        raise ValueError('Invalid repeat/warmup/consecutive count')
    spec = hat_spec(a.run)
    mixed = {**spec, 'runtime_precision': 'float16_safe', 'attention_backend': 'sdpa'}
    qualification = require_runtime_qualification(a.run, mixed, a.precision_checks)
    built = json.loads((a.engine / 'build.json').read_text(encoding='utf-8'))
    graph = validate_graph(Path(built['graph']), a.run)
    scale = graph['display_scale']
    cases = experiment_cases(a.run, scale, a.indices, a.consecutive, a.hard_samples)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'checker_source.py').write_bytes(Path(__file__).read_bytes())
    report = {'status': 'checking', 'passed': False, 'display_scale': scale, 'cases': [],
              'engine_precision': graph['precision'], 'checker_snapshot': 'checker_source.py',
              'protocol_sha256': sha(a.run / 'protocol.json'),
              'engine_sha256': sha(a.engine / 'hat.engine'),
              'build_metadata_sha256': sha(a.engine / 'build.json'),
              'checker_sha256': sha(__file__), 'support_sha256': sha(ROOT / 'display_support.py'),
              'reader_sha256': sha(ROOT / 'ir_sr.py'), 'mixed_qualification': qualification,
              'limits': ERROR_LIMITS, 'temporal_error_limits': TEMPORAL_ERROR_LIMITS,
              'warmup': a.warmup, 'repeats': a.repeats, 'indices': a.indices,
              'consecutive': a.consecutive, 'hard_samples': a.hard_samples,
              'timing_scope': 'prepared CPU grayscale ROI through inference and CPU PIL display resize; excludes reading, model/engine loading, warmup, saving and encoding',
              'memory_note': 'TensorRT native allocations are not captured by PyTorch allocated-memory counters',
              'coverage': 'selected frozen protocol frames and optional hard development samples; not full-video or independent operator acceptance',
              'created_at': utc_now()}
    atomic_json(a.out / 'check.json', report)
    context = {'method': 'fp32', 'phase': 'load'}
    try:
        with run_lock(a.run / 'gpu_inference.lock'):
            model, model_meta = load_model(spec)
            report['reference_metadata'] = model_meta
            report['environment'] = environment()
            for case in cases:
                context = {'method': 'fp32', 'phase': 'warmup', 'sequence': case['sequence'], 'index': case['index']}
                if case['benchmark']:
                    for repeat in range(a.warmup):
                        context['repeat'] = repeat + 1
                        infer(model, spec, case['low'])
                timings, process = [], []
                case.update(fp32_inference_ms=timings, fp32_process_ms=process)
                context['phase'] = 'measure'
                for repeat in range(a.repeats if case['benchmark'] else 1):
                    context['repeat'] = repeat + 1
                    tick = time.perf_counter()
                    native, _, timing = infer(model, spec, case['low'])
                    display = resize(native, (640, 512))
                    process.append((time.perf_counter() - tick) * 1000)
                    timings.append(timing['inference_ms'])
                case.update(ref=native, display=display)
                folder = a.out / 'images' / case['sequence'] / f"frame_{case['index']:05d}"
                save_image(folder / 'fp32.png', display)
                save_image(folder / 'bicubic.png', resize(case['low'], (640, 512)))
            del model
            torch.cuda.empty_cache()
            for method in ('mixed', 'tensorrt'):
                context = {'method': method, 'phase': 'load'}
                if method == 'mixed':
                    model, runtime_meta = load_model(mixed)
                    report['mixed_metadata'] = runtime_meta
                else:
                    runner = EngineRunner(a.engine, a.run)
                previous = {}
                for case in cases:
                    def execute():
                        if method == 'mixed':
                            native, _, timing = infer(model, mixed, case['low'])
                            return native, timing
                        return runner.infer(case['low'])
                    process, native_errors, display_errors, temporal_errors = [], [], [], []
                    row = {'passed': False, 'repeat_native_errors': native_errors,
                           'repeat_display_errors': display_errors, 'runtime_error_changes': temporal_errors,
                           'inference_ms': [], 'process_ms': process, 'gpu_forward_ms': []}
                    case[method] = row
                    context = {'method': method, 'phase': 'warmup', 'sequence': case['sequence'], 'index': case['index']}
                    if case['benchmark']:
                        for repeat in range(a.warmup):
                            context['repeat'] = repeat + 1
                            execute()
                    context['phase'] = 'measure'
                    previous_case = previous.get(case['sequence'])
                    for repeat in range(a.repeats if case['benchmark'] else 1):
                        context['repeat'] = repeat + 1
                        tick = time.perf_counter()
                        native, timing = execute()
                        display = resize(native, (640, 512))
                        process.append((time.perf_counter() - tick) * 1000)
                        row['inference_ms'].append(timing['inference_ms'])
                        if 'gpu_forward_ms' in timing:
                            row['gpu_forward_ms'].append(timing['gpu_forward_ms'])
                        native_errors.append(error_metrics(native, case['ref']))
                        display_errors.append(error_metrics(display, case['display']))
                        delta = display - case['display']
                        if previous_case is not None and previous_case[0] + 1 == case['index']:
                            temporal_errors.append(error_metrics(delta, previous_case[1]))
                    previous[case['sequence']] = (case['index'], delta)
                    passed = all(within_error_limits(e) for e in native_errors + display_errors)
                    passed = passed and all(within_error_limits(e, temporal=True) for e in temporal_errors)
                    row['passed'] = passed
                    save_image(a.out / 'images' / case['sequence'] / f"frame_{case['index']:05d}" / (method + '.png'), display)
                    print(method, scale, case['sequence'], case['index'], passed, flush=True)
                if method == 'mixed':
                    del model
                    torch.cuda.empty_cache()
            report['cases'] = [{k: v for k, v in c.items() if k not in ('low', 'ref', 'display')} for c in cases]
            report['performance'] = {}
            for method in ('fp32', 'mixed', 'tensorrt'):
                summary = {}
                for metric in ('inference_ms', 'process_ms'):
                    values = [v for c in cases if c['benchmark']
                              for v in (c[method + '_' + metric] if method == 'fp32' else c[method][metric])]
                    summary[metric] = {'median': float(np.median(values)),
                                       'p95': float(np.percentile(values, 95)), 'samples': len(values)}
                report['performance'][method] = summary
            report['passed'] = all(c[m]['passed'] for c in cases for m in ('mixed', 'tensorrt'))
            report['status'] = 'completed'
            atomic_json(a.out / 'check.json', report)
            print(json.dumps({'passed': report['passed'], 'cases': len(cases),
                              'performance': report['performance']}, ensure_ascii=False), flush=True)
    except Exception as error:
        report['cases'] = [{k: v for k, v in c.items() if k not in ('low', 'ref', 'display')} for c in cases]
        report.update(status='failed', error=str(error), failure_context=context, updated_at=utc_now())
        atomic_json(a.out / 'check.json', report)
        raise
    if not report['passed']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--run', type=Path, required=True)
    export.add_argument('--input', type=Path, required=True)
    export.add_argument('--scale', type=int, choices=(2, 3), required=True)
    export.add_argument('--roi', type=int, nargs=4)
    export.add_argument('--precision', choices=('float32', 'float16_safe'), default='float32')
    export.add_argument('--out', type=Path, required=True)
    build = sub.add_parser('build')
    build.add_argument('--run', type=Path, required=True)
    build.add_argument('--graph', type=Path, required=True)
    build.add_argument('--out', type=Path, required=True)
    build.add_argument('--workspace-gib', type=int, choices=range(1, 9), default=4)
    build.add_argument('--optimization-level', type=int, choices=range(6), default=3)
    check = sub.add_parser('check')
    check.add_argument('--run', type=Path, required=True)
    check.add_argument('--engine', type=Path, required=True)
    check.add_argument('--out', type=Path, required=True)
    check.add_argument('--precision-checks', type=Path, required=True)
    check.add_argument('--indices', type=int, nargs='+', default=[30, 75, 135])
    check.add_argument('--consecutive', type=int, default=4)
    check.add_argument('--repeats', type=int, default=3)
    check.add_argument('--warmup', type=int, default=1)
    check.add_argument('--hard-samples', action='store_true')
    a = parser.parse_args()
    torch.set_num_threads(4)
    {'export': export_graph, 'build': build_engine, 'check': check_engine}[a.command](a)


if __name__ == '__main__':
    main()
