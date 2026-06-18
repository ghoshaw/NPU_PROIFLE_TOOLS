import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import extract_operators as eo


FA_TYPES = {'FlashAttentionScore', 'FlashAttentionScoreGrad'}
OPERATOR_OUTPUT_FILE = 'interactive_extracted_operators.csv'
STATISTICS_OUTPUT_FILE = 'interactive_operator_statistics.csv'
VECTOR_OUTPUT_FILE = 'interactive_vector_operators.csv'
COMM_OUTPUT_FILE = 'interactive_comm_operators.csv'

TARGET_COMM_PREFIXES = [
    'hcom_allGather_AicpuKernel_',
    'hcom_reduceScatter_AicpuKernel_',
    'hcom_allGather__',
    'hcom_reduceScatter__',
    'hcom_alltoall__',
    'hcom_alltoallv__',
    'hcom_alltoall_AicpuKernel_',
    'hcom_alltoallv_AicpuKernel_',
    'MEMCPY_ASYNC',
]

COMM_HEADERS = ['source_folder', 'name', 'dur', 'data_type', 'count', 'Type']

ADDED_OPERATOR_COLUMNS = [
    'M',
    'K',
    'N',
    'B',
    'N_heads',
    'S_q',
    'S_k',
    'D',
    'SN',
    'causal',
    'FLOPs',
    'MFU',
    'model_runtime(us)',
    'time_ratio(%)',
    eo.RECOMPUTE_STAGE_COL,
    'contribution_to_model_mfu',
    'contribution_to_model_hfu',
    'MBU',
    'I',
    'AI',
    'source_path',
]

STATISTICS_HEADERS = [
    'Type',
    'Duration(us)',
    'Input Shapes',
    'Output Shapes',
    'M',
    'K',
    'N',
    'B',
    'N_heads',
    'S_q',
    'S_k',
    'D',
    'SN',
    'causal',
    'FLOPs',
    'stat_type',
    'MFU',
    'op_count',
    'total_duration(us)',
    'model_runtime(us)',
    'time_ratio(%)',
    eo.RECOMPUTE_STAGE_COL,
    'contribution_to_model_mfu',
    'contribution_to_model_hfu',
    'MBU',
    'I',
    'AI',
]

STAT_REPRESENTATIVE_COLUMNS = [
    'Type',
    'Input Shapes',
    'Output Shapes',
    'M',
    'K',
    'N',
    'B',
    'N_heads',
    'S_q',
    'S_k',
    'D',
    'SN',
    'causal',
    'FLOPs',
    'MBU',
    'I',
    'AI',
]

INTERNAL_COLUMNS = {'__kernel_index', '__global_index'}

VECTOR_ADDED_COLUMNS = [
    'logical_bytes',
    'estimated_bytes',
    'MBU',
    'byte_estimation_rule',
    'model_runtime(us)',
    'time_ratio(%)',
    'source_path',
]

DTYPE_BYTES = {
    'FLOAT': 4,
    'FLOAT32': 4,
    'DT_FLOAT': 4,
    'DOUBLE': 8,
    'FLOAT16': 2,
    'FP16': 2,
    'DT_FLOAT16': 2,
    'DT_BF16': 2,
    'BF16': 2,
    'INT64': 8,
    'UINT64': 8,
    'INT32': 4,
    'UINT32': 4,
    'INT16': 2,
    'UINT16': 2,
    'INT8': 1,
    'UINT8': 1,
    'BOOL': 1,
}

ELEMENTWISE_TYPES = {
    'Add',
    'Sub',
    'Mul',
    'RealDiv',
    'Addcdiv',
    'NotEqual',
    'Greater',
    'Pows',
    'GeluGrad',
    'SiluGrad',
}

MOVE_TYPES = {
    'Cast',
    'Transpose',
    'TransData',
    'TensorMove',
    'Slice',
    'StridedSlice',
    'ConcatD',
    'Pack',
    'RepeatInterleave',
    'Gelu',
    'Swish',
    'RotaryPositionEmbedding',
    'RotaryPositionEmbeddingGrad',
    'MoeTokenUnpermute',
    'MoeTokenUnpermuteGrad',
    'MoeTokenPermuteGrad',
}

OUTPUT_ONLY_TYPES = {
    'ZerosLike',
    'OnesLike',
    'Fill',
    'MemSet',
}


def _round(value: Any) -> Any:
    if isinstance(value, (int, float)) and value != '':
        return round(value, 8)
    return value


def _to_float(value: Any) -> Optional[float]:
    if value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: Optional[float]) -> Any:
    if value is None:
        return ''
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def product(values: List[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def dtype_size(dtype: str) -> int:
    normalized = dtype.strip().strip('"').upper()
    if normalized in {'', 'N/A', 'DT_UNDEFINED'}:
        return 0
    return DTYPE_BYTES.get(normalized, 0)


def split_dtype_string(dtype_str: str) -> List[str]:
    if not dtype_str or dtype_str == 'N/A':
        return []
    return [part.strip().strip('"') for part in dtype_str.split(';')]


def tensor_byte_values(shape_str: str, dtype_str: str) -> List[int]:
    shapes = eo.parse_shape_string(shape_str)
    dtypes = split_dtype_string(dtype_str)
    values = []
    for idx, shape in enumerate(shapes):
        if not shape:
            values.append(0)
            continue
        dtype = dtypes[idx] if idx < len(dtypes) else ''
        values.append(product(shape) * dtype_size(dtype))
    return values


def sum_bytes(values: List[int]) -> int:
    return sum(value for value in values if value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Interactively extract MM/GMM/FA operators from Ascend profiler output.'
    )
    parser.add_argument(
        'profile_path',
        type=Path,
        help='Profile root directory or a single *_pt directory.',
    )
    parser.add_argument(
        '--num',
        type=int,
        default=1,
        help='Number of *_pt folders to include when profile_path is a parent directory.',
    )
    parser.add_argument(
        '--device-flops',
        type=float,
        required=True,
        help='Device compute throughput in TFLOPS, for example A5=432 or A3=354.',
    )
    args = parser.parse_args()
    if args.num < 1:
        parser.error('num must be a positive integer')
    if args.device_flops <= 0:
        parser.error('device-flops must be a positive number')
    return args


def profiler_output_dir(profile_dir: Path) -> Path:
    return profile_dir / 'ASCEND_PROFILER_OUTPUT'


def has_profile_files(profile_dir: Path) -> bool:
    output_dir = profiler_output_dir(profile_dir)
    return (output_dir / 'kernel_details.csv').exists() and (output_dir / 'step_trace_time.csv').exists()


def has_trace_view(profile_dir: Path) -> bool:
    return trace_view_path(profile_dir).exists()


def find_profile_dirs(profile_path: Path, num: int) -> List[Path]:
    if has_profile_files(profile_path):
        return [profile_path]

    profile_dirs = []
    for item in profile_path.iterdir():
        if item.is_dir() and item.name.endswith('_pt') and has_profile_files(item):
            profile_dirs.append(item)
    return profile_dirs[:num]


def find_comm_profile_dirs(profile_path: Path) -> List[Path]:
    if profile_path.name.endswith('_pt') or profiler_output_dir(profile_path).exists():
        return [profile_path]

    profile_dirs = []
    for item in profile_path.iterdir():
        if item.is_dir() and item.name.endswith('_pt'):
            profile_dirs.append(item)
    return profile_dirs


def kernel_details_path(profile_dir: Path) -> Path:
    return profiler_output_dir(profile_dir) / 'kernel_details.csv'


def step_trace_path(profile_dir: Path) -> Path:
    return profiler_output_dir(profile_dir) / 'step_trace_time.csv'


def trace_view_path(profile_dir: Path) -> Path:
    return profiler_output_dir(profile_dir) / 'trace_view.json'


def get_model_runtime(profile_dirs: List[Path]) -> float:
    stage_values = []
    for profile_dir in profile_dirs:
        trace_path = step_trace_path(profile_dir)
        try:
            with trace_path.open(newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    stage = _to_float(row.get('Stage', ''))
                    if stage is not None:
                        stage_values.append(stage)
        except OSError as e:
            print(f"Error reading {trace_path}: {e}")
    return sum(stage_values)


def is_target_operator(type_value: str) -> bool:
    return eo.is_matmul_or_grouped(type_value) or type_value in FA_TYPES


def is_loss_like(row: Dict[str, Any]) -> bool:
    text = f"{row.get('Name', '')} {row.get('Type', '')}".lower()
    return 'loss' in text


def normalize_attention_shape_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    input_tensors = eo.parse_shape_string(row.get('Input Shapes', ''))
    if len(input_tensors) < 2:
        return 'raw', row.get('Input Shapes', '')

    q_shape = input_tensors[0]
    k_shape = input_tensors[1]
    if len(q_shape) == 4 and len(k_shape) == 4:
        B, N, S_q, D = q_shape
        S_k = k_shape[2]
        return 'BNSD', B, N, S_q, S_k, D

    if len(q_shape) == 3 and len(k_shape) == 3:
        T_q, N, D = q_shape
        T_k = k_shape[0]
        return 'TND', T_q, T_k, N, D

    normalized_inputs = tuple(tuple(shape) for shape in input_tensors[:3])
    return 'raw', normalized_inputs


def parse_fa_params(text: str) -> Dict[str, float]:
    parts = [part for part in re.split(r'[\s,]+', text.strip()) if part]
    if len(parts) != 7:
        raise ValueError('expected 7 values: B,N,Sq,Sk,D,SN,causal')

    names = ['B', 'N_heads', 'S_q', 'S_k', 'D', 'SN', 'causal']
    values = {}
    for name, part in zip(names, parts):
        value = float(part)
        if value <= 0:
            raise ValueError(f'{name} must be positive')
        values[name] = value
    return values


def prompt_fa_params(row: Dict[str, Any]) -> Dict[str, float]:
    print("\nNew FA/FAG shape needs parameters:")
    print(f"  Type: {row.get('Type', '')}")
    print(f"  Input Shapes: {row.get('Input Shapes', '')}")
    print(f"  Output Shapes: {row.get('Output Shapes', '')}")
    while True:
        try:
            text = input('Enter B,N,Sq,Sk,D,SN,causal: ')
        except EOFError as e:
            raise SystemExit('Error: missing interactive FA parameters on stdin') from e
        try:
            return parse_fa_params(text)
        except ValueError as e:
            print(f"Invalid FA parameters: {e}")


def calculate_mm_flops_mfu(M: Optional[int],
                           K: Optional[int],
                           N: Optional[int],
                           duration_us: float,
                           device_flops: float) -> Tuple[Optional[float], Optional[float]]:
    if duration_us <= 0 or M is None or K is None or N is None:
        return None, None
    flops = 2.0 * M * K * N
    mfu = flops / device_flops / duration_us / 1_000_000.0
    return flops, mfu


def calculate_mm_mbu(M: Optional[int],
                     K: Optional[int],
                     N: Optional[int],
                     duration_us: float) -> Tuple[Optional[float], Optional[float]]:
    if duration_us <= 0 or M is None or K is None or N is None:
        return None, None
    bytes_count = (M * K + K * N + M * N) * 2
    mbu = bytes_count / 4.0 / duration_us / 1_000_000.0
    return bytes_count, mbu


def calculate_fa_flops_mfu(params: Dict[str, float],
                           op_type: str,
                           duration_us: float,
                           device_flops: float) -> Tuple[Optional[float], Optional[float]]:
    if duration_us <= 0:
        return None, None

    flops = (
        4.0 *
        params['B'] *
        params['S_q'] *
        params['S_k'] *
        params['D'] *
        params['N_heads'] *
        params['SN'] /
        params['causal']
    )
    if op_type == 'FlashAttentionScoreGrad':
        flops *= 2.5
    mfu = flops / device_flops / duration_us / 1_000_000.0
    return flops, mfu


def estimate_vector_bytes(row: Dict[str, Any]) -> Tuple[int, int, str]:
    op_type = row.get('Type', '')
    input_bytes = tensor_byte_values(row.get('Input Shapes', ''), row.get('Input Data Types', ''))
    output_bytes = tensor_byte_values(row.get('Output Shapes', ''), row.get('Output Data Types', ''))
    logical_bytes = sum_bytes(input_bytes) + sum_bytes(output_bytes)

    if op_type == 'LayerNormV3':
        x = input_bytes[0] if len(input_bytes) > 0 else 0
        gamma = input_bytes[1] if len(input_bytes) > 1 else 0
        beta = input_bytes[2] if len(input_bytes) > 2 else 0
        y = output_bytes[0] if len(output_bytes) > 0 else 0
        mean = output_bytes[1] if len(output_bytes) > 1 else 0
        rstd = output_bytes[2] if len(output_bytes) > 2 else 0
        return logical_bytes, 2 * x + y + gamma + beta + mean + rstd, 'layer_norm_two_pass'

    if op_type == 'RmsNorm':
        x = input_bytes[0] if len(input_bytes) > 0 else 0
        weight = input_bytes[1] if len(input_bytes) > 1 else 0
        y = output_bytes[0] if len(output_bytes) > 0 else 0
        rstd = output_bytes[1] if len(output_bytes) > 1 else 0
        return logical_bytes, 2 * x + y + weight + rstd, 'rms_norm_two_pass'

    if op_type in OUTPUT_ONLY_TYPES:
        return logical_bytes, sum_bytes(output_bytes), 'output_only'

    if op_type == 'BroadcastTo':
        first_input = input_bytes[0] if input_bytes else 0
        return logical_bytes, first_input + sum_bytes(output_bytes), 'broadcast_input_output'

    if op_type.startswith('Foreach'):
        return logical_bytes, logical_bytes, 'foreach_logical'

    if op_type in ELEMENTWISE_TYPES:
        return logical_bytes, logical_bytes, 'elementwise_logical'

    if op_type in MOVE_TYPES:
        return logical_bytes, logical_bytes, 'movement_logical'

    return logical_bytes, logical_bytes, 'logical_fallback'


def read_profile_rows(profile_dir: Path) -> Tuple[List[Dict[str, Any]], List[int], List[str]]:
    rows = []
    loss_indices = []
    kernel_path = kernel_details_path(profile_dir)
    with kernel_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for idx, row in enumerate(reader):
            if is_loss_like(row):
                loss_indices.append(idx)
            rows.append(row)
    return rows, loss_indices, headers


def infer_stages_with_loss_anchor(rows: List[Dict[str, Any]], loss_indices: List[int]) -> None:
    eo.infer_recompute_stages(rows, loss_indices)


def enrich_operator_row(row: Dict[str, Any],
                        profile_dir: Path,
                        fa_cache: Dict[Tuple[Any, ...], Dict[str, float]],
                        device_flops: float) -> Dict[str, Any]:
    output_row = dict(row)
    type_value = row.get('Type', '')
    input_shapes = row.get('Input Shapes', '')
    output_shapes = row.get('Output Shapes', '')
    duration_us = _to_float(row.get('Duration(us)', '')) or 0.0

    M = K = N = None
    B = N_heads = S_q = S_k = D = SN = causal = None
    flops = mfu = mbu = intensity = None
    ai = device_flops / 4.0

    if eo.is_matmul_or_grouped(type_value):
        M, K, N = eo.extract_matmul_dims(input_shapes, output_shapes, type_value.startswith('GroupedMatmul'))
        flops, mfu = calculate_mm_flops_mfu(M, K, N, duration_us, device_flops)
        bytes_count, mbu = calculate_mm_mbu(M, K, N, duration_us)
        if flops is not None and bytes_count not in (None, 0):
            intensity = flops / bytes_count
    elif type_value in FA_TYPES:
        key = normalize_attention_shape_key(row)
        if key not in fa_cache:
            fa_cache[key] = prompt_fa_params(row)
        params = fa_cache[key]
        B = params['B']
        N_heads = params['N_heads']
        S_q = params['S_q']
        S_k = params['S_k']
        D = params['D']
        SN = params['SN']
        causal = params['causal']
        flops, mfu = calculate_fa_flops_mfu(params, type_value, duration_us, device_flops)

    output_row['M'] = _format_number(M)
    output_row['K'] = _format_number(K)
    output_row['N'] = _format_number(N)
    output_row['B'] = _format_number(B)
    output_row['N_heads'] = _format_number(N_heads)
    output_row['S_q'] = _format_number(S_q)
    output_row['S_k'] = _format_number(S_k)
    output_row['D'] = _format_number(D)
    output_row['SN'] = _format_number(SN)
    output_row['causal'] = _format_number(causal)
    output_row['FLOPs'] = _round(flops) if flops is not None else ''
    output_row['MFU'] = _round(mfu * 100) if mfu is not None else ''
    output_row['MBU'] = _round(mbu) if mbu is not None else ''
    output_row['I'] = _round(intensity) if intensity is not None else ''
    output_row['AI'] = ai
    output_row['source_path'] = profile_dir.name
    return output_row


def process_profiles(profile_dirs: List[Path],
                     model_runtime: float,
                     device_flops: float) -> Tuple[List[Dict[str, Any]], List[str], int]:
    all_results = []
    fa_cache: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    original_headers: List[str] = []
    global_index = 0

    for profile_dir in profile_dirs:
        source_rows, loss_indices, headers = read_profile_rows(profile_dir)
        if not original_headers and headers:
            original_headers = list(headers)

        extracted_rows = []
        for idx, row in enumerate(source_rows):
            if not is_target_operator(row.get('Type', '')):
                continue
            enriched = enrich_operator_row(row, profile_dir, fa_cache, device_flops)
            enriched['__kernel_index'] = idx
            enriched['__global_index'] = global_index
            global_index += 1
            extracted_rows.append(enriched)

        infer_stages_with_loss_anchor(extracted_rows, loss_indices)
        diagnostics = eo.get_stage_diagnostics(extracted_rows, loss_indices)
        phase_counts = diagnostics['phase_counts']
        print(
            "Stage inference: "
            f"profile={profile_dir.name}, "
            f"forward={phase_counts.get(eo.PHASE_FORWARD, 0)}, "
            f"recompute={phase_counts.get(eo.PHASE_RECOMPUTE, 0)}, "
            f"backward={phase_counts.get(eo.PHASE_BACKWARD, 0)}, "
            f"unknown={phase_counts.get(eo.PHASE_UNKNOWN, 0)}, "
            f"loss_count={diagnostics['loss_count']}, "
            f"first_loss={diagnostics['first_loss_index']}, "
            f"last_loss={diagnostics['last_loss_index']}, "
            f"first_backward_extracted={diagnostics['first_backward_extracted_index']}"
        )
        if phase_counts.get(eo.PHASE_RECOMPUTE, 0) == 0 and diagnostics['loss_count']:
            print(f"WARNING: loss anchors exist but no recompute operators were inferred for {profile_dir.name}")
        for row in extracted_rows:
            phase = row.get(eo.RECOMPUTE_STAGE_COL, eo.PHASE_FORWARD)
            duration_us = _to_float(row.get('Duration(us)', '')) or 0.0
            mfu = _to_float(row.get('MFU', ''))
            time_ratio = _round((duration_us / model_runtime) if model_runtime > 0 else 0)
            contribution = _round(mfu * time_ratio) if mfu is not None else ''

            row['model_runtime(us)'] = model_runtime
            row['time_ratio(%)'] = time_ratio
            row['contribution_to_model_mfu'] = ''
            row['contribution_to_model_hfu'] = ''
            if contribution != '':
                if phase in {eo.PHASE_RECOMPUTE, eo.PHASE_UNKNOWN}:
                    row['contribution_to_model_hfu'] = contribution
                elif phase in {eo.PHASE_FORWARD, eo.PHASE_BACKWARD}:
                    row['contribution_to_model_mfu'] = contribution

        all_results.extend(extracted_rows)

    return all_results, original_headers, len(fa_cache)


def process_vector_profiles(profile_dirs: List[Path],
                            model_runtime: float) -> Tuple[List[Dict[str, Any]], List[str]]:
    vector_rows = []
    original_headers: List[str] = []

    for profile_dir in profile_dirs:
        source_rows, _, headers = read_profile_rows(profile_dir)
        if not original_headers and headers:
            original_headers = list(headers)

        for row in source_rows:
            if row.get('Accelerator Core') != 'AI_VECTOR_CORE':
                continue

            output_row = dict(row)
            duration_us = _to_float(row.get('Duration(us)', '')) or 0.0
            logical_bytes, estimated_bytes, rule = estimate_vector_bytes(row)
            time_ratio = _round((duration_us / model_runtime) if model_runtime > 0 else 0)

            output_row['logical_bytes'] = logical_bytes
            output_row['estimated_bytes'] = estimated_bytes
            output_row['MBU'] = _round(estimated_bytes / (4.0 * duration_us * 1_000_000.0)) if duration_us > 0 else ''
            output_row['byte_estimation_rule'] = rule
            output_row['model_runtime(us)'] = model_runtime
            output_row['time_ratio(%)'] = time_ratio
            output_row['source_path'] = profile_dir.name
            vector_rows.append(output_row)

    return vector_rows, original_headers


def classify_comm_type(name: str, tid: Any) -> Optional[str]:
    if 'Aicpu' in name and 'hcom' in name:
        return 'AICPU'
    if 'hcom' in name:
        return 'CCU'
    if 'MEMCPY_ASYNC' in name:
        try:
            if int(float(tid)) == 56:
                return 'h2d or h2h'
        except (TypeError, ValueError):
            return None
    return None


def process_comm_profiles(profile_dirs: List[Path]) -> List[Dict[str, Any]]:
    comm_rows = []
    for profile_dir in profile_dirs:
        json_path = trace_view_path(profile_dir)
        if not json_path.exists():
            print(f"Skipping communication trace: {json_path}")
            continue

        print(f"Parsing communication trace: {json_path}")
        try:
            with json_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error parsing {json_path}: {e}")
            continue

        if not isinstance(data, list):
            print(f"Skipping communication trace with unexpected JSON root: {json_path}")
            continue

        for obj in data:
            if not isinstance(obj, dict):
                continue

            name = obj.get('name', '')
            if not isinstance(name, str) or not any(name.startswith(prefix) for prefix in TARGET_COMM_PREFIXES):
                continue

            args = obj.get('args') or {}
            if not isinstance(args, dict):
                args = {}

            comm_type = classify_comm_type(name, obj.get('tid', ''))
            if comm_type is None:
                continue

            comm_rows.append({
                'source_folder': profile_dir.name,
                'name': name,
                'dur': obj.get('dur', ''),
                'data_type': args.get('data_type', 'BF16'),
                'count': args.get('count', ''),
                'Type': comm_type,
            })

    return comm_rows


def build_operator_headers(original_headers: List[str]) -> List[str]:
    headers = [h for h in original_headers if h not in ADDED_OPERATOR_COLUMNS and h not in INTERNAL_COLUMNS]
    headers.extend(ADDED_OPERATOR_COLUMNS)
    return headers


def build_vector_headers(original_headers: List[str]) -> List[str]:
    headers = [h for h in original_headers if h not in VECTOR_ADDED_COLUMNS]
    headers.extend(VECTOR_ADDED_COLUMNS)
    return headers


def statistics_group_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    stage = row.get(eo.RECOMPUTE_STAGE_COL, eo.PHASE_FORWARD)
    return (
        row.get('Type', ''),
        row.get('Input Shapes', ''),
        row.get('Output Shapes', ''),
        stage,
    )


def create_stat_row(source_row: Dict[str, Any], stat_type: str) -> Dict[str, Any]:
    row = {h: '' for h in STATISTICS_HEADERS}
    for col in STAT_REPRESENTATIVE_COLUMNS:
        row[col] = source_row.get(col, '')
    row['Duration(us)'] = source_row.get('Duration(us)', '')
    row['MFU'] = source_row.get('MFU', '')
    row['stat_type'] = stat_type
    return row


def compute_statistics(all_results: List[Dict[str, Any]], model_runtime: float) -> Tuple[List[Dict[str, Any]], List[str]]:
    groups = defaultdict(list)
    for row in all_results:
        if row.get('MFU', '') == '':
            continue
        groups[statistics_group_key(row)].append(row)

    group_records = []
    for key, items in sorted(groups.items(), key=lambda item: (eo.get_first_start_time(item[1]), item[0])):
        valid_items = [item for item in items if _to_float(item.get('MFU', '')) is not None]
        if not valid_items:
            continue

        sorted_by_duration = sorted(valid_items, key=lambda item: _to_float(item.get('Duration(us)', '')) or 0.0)
        min_source = sorted_by_duration[0]
        max_source = sorted_by_duration[-1]
        representative = sorted_by_duration[0]
        stage = key[3]

        min_row = create_stat_row(min_source, 'min')
        max_row = create_stat_row(max_source, 'max')

        duration_values = [(_to_float(item.get('Duration(us)', '')) or 0.0) for item in valid_items]
        mfu_values = [(_to_float(item.get('MFU', '')) or 0.0) for item in valid_items]
        op_count = len(valid_items)
        total_duration = _round(sum(duration_values))
        avg_duration = _round(sum(duration_values) / op_count)
        avg_mfu = _round(sum(mfu_values) / op_count)
        time_ratio = _round((total_duration / model_runtime) if model_runtime > 0 else 0)
        contribution = _round(avg_mfu * time_ratio)

        avg_row = {h: '' for h in STATISTICS_HEADERS}
        for col in STAT_REPRESENTATIVE_COLUMNS:
            avg_row[col] = representative.get(col, '')
        avg_row['Duration(us)'] = avg_duration
        avg_row['stat_type'] = 'ave'
        avg_row['MFU'] = avg_mfu
        avg_row['op_count'] = op_count
        avg_row['total_duration(us)'] = total_duration
        avg_row['model_runtime(us)'] = model_runtime
        avg_row['time_ratio(%)'] = time_ratio
        avg_row[eo.RECOMPUTE_STAGE_COL] = stage
        if stage in {eo.PHASE_RECOMPUTE, eo.PHASE_UNKNOWN}:
            avg_row['contribution_to_model_hfu'] = contribution
        elif stage in {eo.PHASE_FORWARD, eo.PHASE_BACKWARD}:
            avg_row['contribution_to_model_mfu'] = contribution

        phase_counts = defaultdict(int)
        for item in valid_items:
            phase_counts[item.get(eo.RECOMPUTE_STAGE_COL, eo.PHASE_FORWARD)] += 1

        group_records.append({
            'key': key,
            'rows': [max_row, min_row, avg_row],
            'representative': representative,
            'stage': stage,
            'forward_signature': eo.get_forward_signature(representative),
            'io_key': eo.get_matmul_io_key(representative),
            'phase_counts': dict(phase_counts),
            'op_count': op_count,
            'first_start': eo.get_first_start_time(valid_items),
        })

    ordered_records = eo.order_statistics_group_records(group_records)
    stat_rows = []
    for record in ordered_records:
        stat_rows.extend(record['rows'])

    return stat_rows, STATISTICS_HEADERS


def write_csv(path: Path, headers: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    profile_dirs = find_profile_dirs(args.profile_path, args.num)
    if not profile_dirs:
        raise SystemExit(f"Error: no profile folders with ASCEND_PROFILER_OUTPUT found in {args.profile_path}")

    output_dir = args.profile_path
    operator_output_file = output_dir / OPERATOR_OUTPUT_FILE
    statistics_output_file = output_dir / STATISTICS_OUTPUT_FILE
    vector_output_file = output_dir / VECTOR_OUTPUT_FILE
    comm_output_file = output_dir / COMM_OUTPUT_FILE

    print(f"Found {len(profile_dirs)} profile folder(s)")
    for profile_dir in profile_dirs:
        print(f"Processing profile folder: {profile_dir.name}")

    model_runtime = get_model_runtime(profile_dirs)
    print(f"Model runtime Stage sum: {model_runtime:.2f} us")
    print(f"Device FLOPS: {args.device_flops}")

    all_results, original_headers, fa_prompt_count = process_profiles(profile_dirs, model_runtime, args.device_flops)
    if not all_results:
        raise SystemExit('No matching operators found')

    operator_headers = build_operator_headers(original_headers)
    stat_rows, stat_headers = compute_statistics(all_results, model_runtime)
    vector_rows, vector_original_headers = process_vector_profiles(profile_dirs, model_runtime)
    vector_headers = build_vector_headers(vector_original_headers)
    comm_profile_dirs = find_comm_profile_dirs(args.profile_path)
    comm_rows = process_comm_profiles(comm_profile_dirs)

    write_csv(operator_output_file, operator_headers, all_results)
    write_csv(statistics_output_file, stat_headers, stat_rows)
    write_csv(vector_output_file, vector_headers, vector_rows)
    write_csv(comm_output_file, COMM_HEADERS, comm_rows)

    matmul_count = sum(1 for row in all_results if row.get('Type', '').startswith('MatMulV') or
                       (row.get('Type', '').startswith('MatMul') and not row.get('Type', '').startswith('GroupedMatmul')))
    grouped_count = sum(1 for row in all_results if row.get('Type', '').startswith('GroupedMatmul'))
    fa_count = sum(1 for row in all_results if row.get('Type', '') == 'FlashAttentionScore')
    fa_grad_count = sum(1 for row in all_results if row.get('Type', '') == 'FlashAttentionScoreGrad')

    print(f"\nOperator results written to: {operator_output_file}")
    print(f"Statistics results written to: {statistics_output_file}")
    print(f"Vector results written to: {vector_output_file}")
    print(f"Communication results written to: {comm_output_file}")
    print(f"Operator rows: {len(all_results)}")
    print(f"Statistics rows: {len(stat_rows)}")
    print(f"Vector rows: {len(vector_rows)}")
    print(f"Communication profile folders scanned: {len(comm_profile_dirs)}")
    print(f"Communication rows: {len(comm_rows)}")
    print(f"FA/FAG shape prompts answered: {fa_prompt_count}")
    print("\nOperator breakdown:")
    print(f"  MatMul*: {matmul_count}")
    print(f"  GroupedMatmul*: {grouped_count}")
    print(f"  FlashAttentionScore: {fa_count}")
    print(f"  FlashAttentionScoreGrad: {fa_grad_count}")

    warnings = eo.validate_matmul_backward_counts(all_results)
    print("\nMM/GMM backward count check:")
    if warnings:
        print(f"  WARNING: {len(warnings)} shape(s) do not match one logical forward to two backward ops")
        for warning in warnings[:20]:
            print(f"    {warning}")
        if len(warnings) > 20:
            print(f"    ... {len(warnings) - 20} more")
    else:
        print("  OK: every MM/GMM with backward matches one logical forward to two backward ops")


if __name__ == '__main__':
    main()
