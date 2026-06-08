import os
import csv
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from collections import defaultdict
import statistics

RECOMPUTE_STAGE_COL = 'recompute_stage'
PHASE_FORWARD = 'forward'
PHASE_RECOMPUTE = 'recompute'
PHASE_BACKWARD = 'backward'


def parse_shape_string(shape_str: str) -> List[List[int]]:
    if not shape_str or shape_str == 'N/A':
        return []

    shape_str = shape_str.strip().strip('"')
    tensors = []

    parts = shape_str.split(';')
    for part in parts:
        part = part.strip().strip('"')
        if not part:
            tensors.append([])
            continue

        if part.startswith('"""') and part.endswith('"""'):
            part = part[3:-3]

        numbers = re.findall(r'\d+', part)
        if numbers:
            tensors.append([int(n) for n in numbers])
        else:
            tensors.append([])

    return tensors


def extract_matmul_dims(input_shapes: str, output_shapes: str, is_grouped: bool = False) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    input_tensors = parse_shape_string(input_shapes)
    output_tensors = parse_shape_string(output_shapes)

    if not input_tensors or len(input_tensors) < 2:
        return None, None, None

    if not output_tensors:
        return None, None, None

    output = output_tensors[0]
    if len(output) < 2:
        return None, None, None

    M = output[0]
    N = output[-1]

    if is_grouped:
        first_input = input_tensors[0]
        M = first_input[0]
        K = first_input[1]
        return M, K, N

    first_input = input_tensors[0]
    second_input = input_tensors[1]

    if first_input[0] == output[0]:
        K = first_input[1]
        return M, K, N
    elif first_input[1] == output[0]:
        K = first_input[0]
        return M, K, N

    return None, None, None


def extract_fa_dims(input_shapes: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    input_tensors = parse_shape_string(input_shapes)

    if not input_tensors:
        return None, None, None, None, None

    first_tensor = input_tensors[0]

    if len(first_tensor) == 4:
        B, N, S, D = first_tensor[0], first_tensor[1], first_tensor[2], first_tensor[3]
        S_key = input_tensors[1][2] if len(input_tensors) > 1 and len(input_tensors[1]) > 2 else S
        return B, N, S, D, S_key
    elif len(first_tensor) == 3:
        return 1, first_tensor[1], first_tensor[0], first_tensor[2], first_tensor[0]

    return None, None, None, None, None


def calculate_flops_mfu(M: Optional[int], K: Optional[int], N: Optional[int],
                        B: Optional[int], S: Optional[int], S_key: Optional[int], D: Optional[int],
                        op_type: str, duration_us: float) -> Tuple[Optional[float], Optional[float]]:
    if duration_us <= 0:
        return None, None

    flops = None

    if op_type.startswith('MatMul') or op_type.startswith('GroupedMatmul'):
        if M is not None and K is not None and N is not None:
            flops = 2.0 * M * N * K
    elif op_type == 'FlashAttentionScore':
        if B is not None and S is not None and D is not None:
            N_val = N if N is not None else 1
            flops = 4.0 * B * S * S_key * N_val * D
    elif op_type == 'FlashAttentionScoreGrad':
        if B is not None and S is not None and D is not None:
            N_val = N if N is not None else 1
            flops = 4.0 * B * S * S_key * N_val * D * 2.5

    if flops is not None:
        mfu = flops / (432.0 * duration_us * 1_000_000.0)
        return flops, mfu

    return None, None


def calculate_flops_mbu(M: Optional[int], K: Optional[int], N: Optional[int],
                        B: Optional[int], S: Optional[int], S_key: Optional[int], D: Optional[int],
                        op_type: str, duration_us: float) -> Tuple[Optional[float], Optional[float]]:
    if duration_us <= 0:
        return None, None

    flops = None

    if op_type.startswith('MatMul') or op_type.startswith('GroupedMatmul'):
        if M is not None and K is not None and N is not None:
            flops = (M * K + K * N + M * N) * 2

    if flops is not None:
        mbu = flops / (4.0 * duration_us * 1_000_000.0)
        return flops, mbu

    return None, None


def is_matmul_or_grouped(type_val: str) -> bool:
    is_grouped = type_val.startswith('GroupedMatmul')
    is_matmul = (type_val.startswith('MatMulV') or
                (type_val.startswith('MatMul') and not is_grouped))
    return is_matmul or is_grouped


def get_exact_shape_key(row: Dict[str, Any]) -> str:
    return f"{row.get('Type', '')}|{row.get('Input Shapes', '')}|{row.get('Output Shapes', '')}"


def is_weight_grad_like(row: Dict[str, Any]) -> bool:
    type_val = row.get('Type', '')
    if not is_matmul_or_grouped(type_val):
        return False

    input_tensors = parse_shape_string(row.get('Input Shapes', ''))
    output_tensors = parse_shape_string(row.get('Output Shapes', ''))
    if len(input_tensors) < 2 or not output_tensors:
        return False

    first_input = input_tensors[0]
    second_input = input_tensors[1]
    output = output_tensors[0]
    if not first_input or not second_input or not output:
        return False

    same_activation_axis = first_input[0] == second_input[0]
    if type_val.startswith('GroupedMatmul'):
        return same_activation_axis and len(output) >= 3

    return same_activation_axis and len(output) >= 2 and output[0] != first_input[0]


def infer_recompute_stages(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    for row in rows:
        row[RECOMPUTE_STAGE_COL] = PHASE_FORWARD

    key_to_indices = defaultdict(list)
    for idx, row in enumerate(rows):
        key_to_indices[get_exact_shape_key(row)].append(idx)

    recompute_anchor_indices = []
    for indices in key_to_indices.values():
        first_row = rows[indices[0]]
        if first_row.get('Type', '') != 'FlashAttentionScore' or len(indices) < 2:
            continue
        split_idx = len(indices) // 2
        recompute_anchor_indices.extend(indices[split_idx:])

    if not recompute_anchor_indices:
        first_backward_idx = next(
            (idx for idx, row in enumerate(rows)
             if row.get('Type', '') == 'FlashAttentionScoreGrad' or is_weight_grad_like(row)),
            None
        )
        if first_backward_idx is not None:
            for idx in range(first_backward_idx, len(rows)):
                type_val = rows[idx].get('Type', '')
                if is_matmul_or_grouped(type_val) or type_val == 'FlashAttentionScoreGrad':
                    rows[idx][RECOMPUTE_STAGE_COL] = PHASE_BACKWARD
            refine_backward_stages_by_shape(rows)
            classify_matmul_recompute_by_backward_counts(rows)
        return

    first_recompute_anchor = min(recompute_anchor_indices)
    forward_recompute_keys = {
        key for key, indices in key_to_indices.items()
        if indices[0] < first_recompute_anchor and indices[-1] >= first_recompute_anchor
    }

    def is_forward_recompute_candidate(idx: int) -> bool:
        row = rows[idx]
        type_val = row.get('Type', '')
        if type_val == 'FlashAttentionScoreGrad' or is_weight_grad_like(row):
            return False
        if not (is_matmul_or_grouped(type_val) or type_val == 'FlashAttentionScore'):
            return False
        return get_exact_shape_key(row) in forward_recompute_keys

    segments = []
    for anchor_idx in sorted(recompute_anchor_indices):
        start = anchor_idx
        while start > 0 and is_forward_recompute_candidate(start - 1):
            start -= 1

        end = anchor_idx
        while end + 1 < len(rows) and is_forward_recompute_candidate(end + 1):
            end += 1

        segments.append((start, end))

    merged_segments = []
    for start, end in sorted(segments):
        if not merged_segments or start > merged_segments[-1][1] + 1:
            merged_segments.append([start, end])
        else:
            merged_segments[-1][1] = max(merged_segments[-1][1], end)

    for start, end in merged_segments:
        for idx in range(start, end + 1):
            if is_forward_recompute_candidate(idx):
                rows[idx][RECOMPUTE_STAGE_COL] = PHASE_RECOMPUTE

    first_recompute_start = min(start for start, _ in merged_segments)
    for idx, row in enumerate(rows):
        if row.get(RECOMPUTE_STAGE_COL) == PHASE_RECOMPUTE:
            continue
        type_val = row.get('Type', '')
        if type_val == 'FlashAttentionScoreGrad':
            row[RECOMPUTE_STAGE_COL] = PHASE_BACKWARD
        elif is_matmul_or_grouped(type_val) and (idx >= first_recompute_start or is_weight_grad_like(row)):
            row[RECOMPUTE_STAGE_COL] = PHASE_BACKWARD

    refine_backward_stages_by_shape(rows)
    classify_matmul_recompute_by_backward_counts(rows)


def refine_backward_stages_by_shape(rows: List[Dict[str, Any]]) -> None:
    forward_indices_by_io_key = defaultdict(list)
    for idx, row in enumerate(rows):
        if not is_matmul_or_grouped(row.get('Type', '')) or row.get(RECOMPUTE_STAGE_COL) == PHASE_BACKWARD:
            continue

        io_key = get_matmul_io_key(row)
        if io_key is not None:
            forward_indices_by_io_key[io_key].append(idx)

    forward_io_keys = set(forward_indices_by_io_key.keys())

    for idx, row in enumerate(rows):
        if row.get(RECOMPUTE_STAGE_COL) != PHASE_BACKWARD or not is_weight_grad_like(row):
            continue

        dw_matches = []
        for candidate_io_key in forward_io_keys:
            role = get_backward_role_for_io_key(row, candidate_io_key)
            if role == 'dw':
                dw_matches.append(candidate_io_key)

        for io_key in dw_matches:
            for prev_idx in range(idx - 1, -1, -1):
                prev_row = rows[prev_idx]
                if (not is_matmul_or_grouped(prev_row.get('Type', '')) or
                        prev_row.get(RECOMPUTE_STAGE_COL) == PHASE_BACKWARD or
                        is_weight_grad_like(prev_row)):
                    continue

                prev_io_key, prev_role = choose_backward_io_match(prev_row, {io_key})
                has_earlier_forward = any(forward_idx < prev_idx for forward_idx in forward_indices_by_io_key[io_key])
                if prev_io_key == io_key and prev_role == 'dx' and has_earlier_forward:
                    prev_row[RECOMPUTE_STAGE_COL] = PHASE_BACKWARD
                    break


def classify_matmul_recompute_by_backward_counts(rows: List[Dict[str, Any]]) -> None:
    candidate_forward_keys = set()
    for row in rows:
        if (not is_matmul_or_grouped(row.get('Type', '')) or
                row.get(RECOMPUTE_STAGE_COL) == PHASE_BACKWARD or
                is_weight_grad_like(row)):
            continue

        io_key = get_matmul_io_key(row)
        if io_key is not None:
            candidate_forward_keys.add(io_key)

    if not candidate_forward_keys:
        return

    backward_counts = defaultdict(lambda: defaultdict(int))
    matched_backward_indices = set()
    for idx, row in enumerate(rows):
        if not is_matmul_or_grouped(row.get('Type', '')) or row.get(RECOMPUTE_STAGE_COL) != PHASE_BACKWARD:
            continue

        io_key, role = choose_backward_io_match(row, candidate_forward_keys)
        if io_key is None or role is None:
            continue

        backward_counts[io_key][role] += 1
        matched_backward_indices.add(idx)

    forward_like_indices = defaultdict(list)
    for idx, row in enumerate(rows):
        if idx in matched_backward_indices:
            continue
        if (not is_matmul_or_grouped(row.get('Type', '')) or
                row.get(RECOMPUTE_STAGE_COL) == PHASE_BACKWARD or
                is_weight_grad_like(row)):
            continue

        io_key = get_matmul_io_key(row)
        if io_key is not None:
            forward_like_indices[io_key].append(idx)

    for io_key, indices in forward_like_indices.items():
        dx_count = backward_counts[io_key].get('dx', 0)
        dw_count = backward_counts[io_key].get('dw', 0)
        if dx_count == 0 or dw_count == 0:
            continue

        logical_forward_count = min(dx_count, dw_count)
        if len(indices) <= logical_forward_count:
            continue

        for idx in indices[:logical_forward_count]:
            if rows[idx].get(RECOMPUTE_STAGE_COL) != PHASE_BACKWARD:
                rows[idx][RECOMPUTE_STAGE_COL] = PHASE_FORWARD
        for idx in indices[logical_forward_count:]:
            rows[idx][RECOMPUTE_STAGE_COL] = PHASE_RECOMPUTE


def process_kernel_details(csv_path: Path) -> List[Dict[str, Any]]:
    results = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        if not headers:
            return results

        for row in reader:
            type_val = row.get('Type', '')

            is_grouped = type_val.startswith('GroupedMatmul')
            is_matmul = (type_val.startswith('MatMulV') or
                        (type_val.startswith('MatMul') and not is_grouped))
            is_fa = type_val == 'FlashAttentionScore'
            is_fa_grad = type_val == 'FlashAttentionScoreGrad'

            if not (is_matmul or is_grouped or is_fa or is_fa_grad):
                continue

            input_shapes = row.get('Input Shapes', '')
            output_shapes = row.get('Output Shapes', '')
            duration_str = row.get('Duration(us)', '0')

            try:
                duration_us = float(duration_str)
            except (ValueError, TypeError):
                duration_us = 0.0

            M, K, N = None, None, None
            B, N_head, S, S_key, D = None, None, None, None, None
            flops, flops_bw, mfu, mbu, I = None, None, None, None, None

            if is_matmul or is_grouped:
                M, K, N = extract_matmul_dims(input_shapes, output_shapes, is_grouped)
                flops, mfu = calculate_flops_mfu(M, K, N, None, None, None, None, type_val, duration_us)
                flops_bw, mbu = calculate_flops_mbu(M, K, N, None, None, None, None, type_val, duration_us)
                I = flops / flops_bw
            elif is_fa or is_fa_grad:
                B, N_head, S, D, S_key = extract_fa_dims(input_shapes)
                flops, mfu = calculate_flops_mfu(None, None, N_head, B, S, S_key, D, type_val, duration_us)

            result_row = dict(row)
            result_row['M'] = M if M is not None else ''
            result_row['K'] = K if K is not None else ''
            result_row['N'] = N if N is not None else ''
            result_row['B'] = B if B is not None else ''
            result_row['N_heads'] = N_head if N_head is not None else ''
            result_row['S_q'] = S if S is not None else ''
            result_row['S_k'] = S_key if S_key is not None else ''
            result_row['D'] = D if D is not None else ''
            result_row['FLOPs'] = _round(flops) if flops is not None else ''
            result_row['MFU'] = _round(mfu*100) if mfu is not None else ''
            result_row['MBU'] = _round(mbu) if mbu is not None else ''
            result_row['I'] = _round(I) if I is not None else ''
            result_row['AI'] = 432/4.0
            result_row['source_path'] = csv_path.parent.parent.name if csv_path else ''

            results.append(result_row)

    infer_recompute_stages(results)
    return results


def find_ascend_pt_folders(base_dir: Path) -> List[Path]:
    folders = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.endswith('_ascend_pt'):
            profiler_output = item / 'ASCEND_PROFILER_OUTPUT'
            if profiler_output.exists():
                kernel_csv = profiler_output / 'kernel_details.csv'
                if kernel_csv.exists():
                    folders.append(kernel_csv)
    return folders


def get_model_runtime(base_dir: Path) -> float:
    stage_values = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.endswith('_ascend_pt'):
            profiler_output = item / 'ASCEND_PROFILER_OUTPUT'
            if profiler_output.exists():
                step_trace_file = profiler_output / 'step_trace_time.csv'
                if step_trace_file.exists():
                    try:
                        with open(step_trace_file, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                stage_str = row.get('Stage', '')
                                if stage_str:
                                    try:
                                        stage_values.append(float(stage_str))
                                    except (ValueError, TypeError):
                                        pass
                    except Exception as e:
                        print(f"Error reading {step_trace_file}: {e}")

    if stage_values:
        return sum(stage_values)
    return 0.0


def get_group_key(row: Dict[str, Any]) -> str:
    type_val = row.get('Type', '')
    if type_val.startswith('MatMul') or type_val.startswith('GroupedMatmul'):
        shape_key = f"input={row.get('Input Shapes', '')},out={row.get('Output Shapes', '')}"
    elif type_val.startswith('FlashAttention'):
        shape_key = f"B={row.get('B', '')},N={row.get('N_heads', '')},S_q={row.get('S_q', '')},S_k={row.get('S_k', '')},D={row.get('D', '')}"
    else:
        shape_key = ''
    return f"{type_val}|{shape_key}"


def get_statistics_group_key(row: Dict[str, Any]) -> str:
    phase = row.get(RECOMPUTE_STAGE_COL, PHASE_FORWARD)
    if phase not in {PHASE_FORWARD, PHASE_RECOMPUTE, PHASE_BACKWARD}:
        phase = PHASE_FORWARD
    return f"{get_group_key(row)}|{phase}"


def get_matmul_family(type_val: str) -> str:
    if type_val.startswith('GroupedMatmul'):
        return 'GroupedMatmul'
    if type_val.startswith('MatMul'):
        return 'MatMul'
    return type_val


def _to_int(value: Any) -> Optional[int]:
    if value == '':
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def get_forward_signature(row: Dict[str, Any]) -> Optional[Tuple[str, int, int, int]]:
    type_val = row.get('Type', '')
    if not is_matmul_or_grouped(type_val):
        return None

    M = _to_int(row.get('M', ''))
    K = _to_int(row.get('K', ''))
    N = _to_int(row.get('N', ''))
    if M is None or K is None or N is None:
        return None

    return get_matmul_family(type_val), M, K, N


def _shape_tuple(shape: List[int]) -> Tuple[int, ...]:
    return tuple(shape)


def get_matmul_io_key(row: Dict[str, Any]) -> Optional[Tuple[str, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
    type_val = row.get('Type', '')
    if not is_matmul_or_grouped(type_val):
        return None

    input_tensors = parse_shape_string(row.get('Input Shapes', ''))
    output_tensors = parse_shape_string(row.get('Output Shapes', ''))
    if len(input_tensors) < 2 or not output_tensors:
        return None

    activation_shape = input_tensors[0]
    weight_shape = input_tensors[1]
    output_shape = output_tensors[0]
    if not activation_shape or not weight_shape or not output_shape:
        return None

    return (
        get_matmul_family(type_val),
        _shape_tuple(activation_shape),
        _shape_tuple(weight_shape),
        _shape_tuple(output_shape),
    )


def get_backward_role_for_io_key(row: Dict[str, Any],
                                 io_key: Tuple[str, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]) -> Optional[str]:
    role, _ = get_backward_role_and_priority_for_io_key(row, io_key)
    return role


def get_backward_role_and_priority_for_io_key(row: Dict[str, Any],
                                              io_key: Tuple[str, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]) -> Tuple[Optional[str], int]:
    type_val = row.get('Type', '')
    if not is_matmul_or_grouped(type_val):
        return None, 99

    family, activation_shape, weight_shape, output_shape = io_key
    if get_matmul_family(type_val) != family:
        return None, 99

    input_tensors = [_shape_tuple(t) for t in parse_shape_string(row.get('Input Shapes', '')) if t]
    output_tensors = [_shape_tuple(t) for t in parse_shape_string(row.get('Output Shapes', '')) if t]
    if not input_tensors or not output_tensors:
        return None, 99

    grad_output_shape = output_tensors[0]
    has_forward_output = output_shape in input_tensors
    has_activation = activation_shape in input_tensors
    has_weight = weight_shape in input_tensors

    if not is_weight_grad_like(row):
        if grad_output_shape == activation_shape and has_forward_output and has_weight:
            return 'dx', 0
        return None, 99

    if grad_output_shape == weight_shape and has_activation and has_forward_output:
        return 'dw', 0
    if family == 'MatMul' and len(weight_shape) == 2 and grad_output_shape == (weight_shape[1], weight_shape[0]) and has_activation and has_forward_output:
        return 'dw', 1
    return None, 99


def choose_backward_io_match(row: Dict[str, Any],
                             forward_io_keys: set) -> Tuple[Optional[Tuple[str, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]], Optional[str]]:
    matches = []
    for io_key in forward_io_keys:
        role, priority = get_backward_role_and_priority_for_io_key(row, io_key)
        if role is not None:
            matches.append((io_key, role, priority))

    if len(matches) == 1:
        io_key, role, _ = matches[0]
        return io_key, role
    if not matches:
        return None, None

    role_order = {'dx': 0, 'dw': 1}
    matches.sort(key=lambda item: (item[2], role_order[item[1]], item[0][0], item[0][1], item[0][2], item[0][3]))
    io_key, role, _ = matches[0]
    return io_key, role


def describe_io_key(io_key: Tuple[str, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]) -> str:
    family, activation_shape, weight_shape, output_shape = io_key
    return f"{family}(A={activation_shape}, W={weight_shape}, Out={output_shape})"


def get_first_start_time(items: List[Dict[str, Any]]) -> float:
    start_values = []
    for item in items:
        start_str = str(item.get('Start Time(us)', '')).strip()
        try:
            start_values.append(float(start_str))
        except (ValueError, TypeError):
            pass
    return min(start_values) if start_values else 0.0


def get_matmul_relationships(group_records: List[Dict[str, Any]]) -> Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]], Dict[str, Any]]:
    forward_io_keys = {
        record['io_key']
        for record in group_records
        if record.get('io_key') is not None and record.get('stage') == PHASE_FORWARD
    }

    relationships = defaultdict(lambda: {
        'forward_records': [],
        'recompute_records': [],
        'backward_records_by_role': defaultdict(list),
        'logical_forward_count': 0,
    })

    for record in group_records:
        row = record['representative']
        type_val = row.get('Type', '')
        if not is_matmul_or_grouped(type_val):
            continue

        record['backward_role'] = None
        if record['stage'] == PHASE_BACKWARD:
            io_key, role = choose_backward_io_match(row, forward_io_keys)
            record['io_key'] = io_key
            record['backward_role'] = role
            if io_key is not None and role is not None:
                relationships[io_key]['backward_records_by_role'][role].append(record)
        elif record['stage'] == PHASE_FORWARD and record.get('io_key') is not None:
            io_key = record['io_key']
            relationships[io_key]['forward_records'].append(record)
            relationships[io_key]['logical_forward_count'] += record.get('phase_counts', {}).get(PHASE_FORWARD, 0)
        elif record['stage'] == PHASE_RECOMPUTE and record.get('io_key') is not None:
            relationships[record['io_key']]['recompute_records'].append(record)

    for relation in relationships.values():
        relation['forward_records'].sort(key=lambda r: r['first_start'])
        relation['recompute_records'].sort(key=lambda r: r['first_start'])
        for role_records in relation['backward_records_by_role'].values():
            role_records.sort(key=lambda r: r['first_start'])

        dx_records = relation['backward_records_by_role'].get('dx', [])
        dw_records = relation['backward_records_by_role'].get('dw', [])
        logical_forward_count = relation['logical_forward_count']
        relation['is_complete'] = (
            bool(relation['forward_records']) and
            len(dx_records) == 1 and
            len(dw_records) == 1 and
            dx_records[0]['op_count'] == logical_forward_count and
            dw_records[0]['op_count'] == logical_forward_count
        )

    return relationships


def is_complete_matmul_stat_record(record: Dict[str, Any], complete_io_keys: set) -> bool:
    io_key = record.get('io_key')
    if io_key not in complete_io_keys:
        return False
    if record.get('stage') == PHASE_FORWARD:
        return True
    return record.get('stage') == PHASE_BACKWARD and record.get('backward_role') in {'dx', 'dw'}


def get_backward_parent_candidates(row: Dict[str, Any]) -> List[Tuple[str, int, int, int]]:
    type_val = row.get('Type', '')
    if not is_matmul_or_grouped(type_val):
        return []

    family = get_matmul_family(type_val)
    input_tensors = parse_shape_string(row.get('Input Shapes', ''))
    output_tensors = parse_shape_string(row.get('Output Shapes', ''))
    candidates = []

    if is_weight_grad_like(row) and len(input_tensors) >= 2 and output_tensors:
        first_input = input_tensors[0]
        second_input = input_tensors[1]
        output = output_tensors[0]
        if first_input and second_input and len(first_input) >= 2 and len(second_input) >= 2:
            M = first_input[0]
            if type_val.startswith('GroupedMatmul') and len(output) >= 3:
                candidates.append((family, M, output[-2], output[-1]))
            elif len(output) >= 2:
                candidates.append((family, M, output[-2], output[-1]))
                candidates.append((family, M, output[-1], output[-2]))

            candidates.append((family, M, first_input[1], second_input[1]))
            candidates.append((family, M, second_input[1], first_input[1]))
    else:
        M = _to_int(row.get('M', ''))
        K = _to_int(row.get('K', ''))
        N = _to_int(row.get('N', ''))
        if M is not None and K is not None and N is not None:
            candidates.append((family, M, N, K))

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def choose_backward_parent_signature(row: Dict[str, Any],
                                     forward_signatures: set) -> Optional[Tuple[str, int, int, int]]:
    candidates = get_backward_parent_candidates(row)
    for candidate in candidates:
        if candidate in forward_signatures:
            return candidate
    return candidates[0] if candidates else None


def order_statistics_group_records(group_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    relationships = get_matmul_relationships(group_records)
    ordered_records = []
    incomplete_matmul_records = []
    emitted_ids = set()
    emitted_complete_io_keys = set()
    sorted_records = sorted(group_records, key=lambda r: (r['first_start'], r['key']))

    def emit_complete_matmul(io_key: Tuple[str, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]) -> None:
        relation = relationships[io_key]
        complete_records = (
            relation['forward_records'] +
            relation['backward_records_by_role']['dx'] +
            relation['backward_records_by_role']['dw']
        )
        for related_record in complete_records:
            related_id = id(related_record)
            if related_id not in emitted_ids:
                ordered_records.append(related_record)
                emitted_ids.add(related_id)
        emitted_complete_io_keys.add(io_key)

    for record in sorted_records:
        record_id = id(record)
        if record_id in emitted_ids:
            continue

        row = record['representative']
        type_val = row.get('Type', '')
        if not is_matmul_or_grouped(type_val):
            ordered_records.append(record)
            emitted_ids.add(record_id)
            continue

        if record.get('stage') == PHASE_RECOMPUTE:
            ordered_records.append(record)
            emitted_ids.add(record_id)
            continue

        io_key = record.get('io_key')
        relation = relationships.get(io_key) if io_key is not None else None
        if relation is not None and relation.get('is_complete'):
            if io_key not in emitted_complete_io_keys and record.get('stage') == PHASE_FORWARD:
                emit_complete_matmul(io_key)
            continue

        incomplete_matmul_records.append(record)
        emitted_ids.add(record_id)

    for record in sorted_records:
        if id(record) not in emitted_ids:
            row = record['representative']
            if is_matmul_or_grouped(row.get('Type', '')):
                if record.get('stage') == PHASE_RECOMPUTE:
                    ordered_records.append(record)
                    emitted_ids.add(id(record))
                    continue

                io_key = record.get('io_key')
                relation = relationships.get(io_key) if io_key is not None else None
                if relation is not None and relation.get('is_complete') and io_key not in emitted_complete_io_keys:
                    emit_complete_matmul(io_key)
                elif id(record) not in emitted_ids:
                    incomplete_matmul_records.append(record)
                    emitted_ids.add(id(record))
            else:
                ordered_records.append(record)
                emitted_ids.add(id(record))

    ordered_records.extend(incomplete_matmul_records)

    return ordered_records


def validate_matmul_backward_counts(all_results: List[Dict[str, Any]]) -> List[str]:
    forward_counts = defaultdict(int)
    recompute_counts = defaultdict(int)
    backward_counts = defaultdict(lambda: defaultdict(int))

    for row in all_results:
        type_val = row.get('Type', '')
        if not is_matmul_or_grouped(type_val):
            continue

        phase = row.get(RECOMPUTE_STAGE_COL, PHASE_FORWARD)
        if phase == PHASE_BACKWARD:
            continue

        io_key = get_matmul_io_key(row)
        if io_key is None:
            continue

        if phase == PHASE_RECOMPUTE:
            recompute_counts[io_key] += 1
        else:
            forward_counts[io_key] += 1

    forward_io_keys = set(forward_counts.keys())

    for row in all_results:
        type_val = row.get('Type', '')
        if not is_matmul_or_grouped(type_val) or row.get(RECOMPUTE_STAGE_COL) != PHASE_BACKWARD:
            continue

        io_key, role = choose_backward_io_match(row, forward_io_keys)
        if io_key is not None and role is not None:
            backward_counts[io_key][role] += 1

    warnings = []
    for io_key in sorted(forward_io_keys):
        forward_count = forward_counts.get(io_key, 0)
        if forward_count == 0:
            continue

        dx_count = backward_counts[io_key].get('dx', 0)
        dw_count = backward_counts[io_key].get('dw', 0)
        if dx_count != forward_count or dw_count != forward_count:
            warnings.append(
                f"{describe_io_key(io_key)} forward={forward_count:g}, recompute={recompute_counts.get(io_key, 0):g}, "
                f"dx={dx_count:g}, dw={dw_count:g}, expected_dx=expected_dw={forward_count:g}"
            )

    return warnings


def create_empty_row(headers: List[str]) -> Dict[str, str]:
    return {h: '' for h in headers}


def _round(v):
    """保留4位小数"""
    if isinstance(v, (int, float)) and v != '':
        return round(v, 4)
    return v

def compute_statistics(all_results: List[Dict[str, Any]], headers: List[str], model_runtime: float) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for row in all_results:
        key = get_statistics_group_key(row)
        if key:
            groups[key].append(row)

    new_rows = []
    new_col_name = 'stat_type'
    headers = [h for h in headers if h != RECOMPUTE_STAGE_COL]

    if 'source_path' in headers:
        source_idx = headers.index('MFU')
        new_headers = headers[:source_idx] + [new_col_name] + headers[source_idx:]
    else:
        new_headers = [new_col_name] + headers

    group_records = []
    sorted_groups = sorted(groups.items(), key=lambda item: (get_first_start_time(item[1]), item[0]))
    for key, items in sorted_groups:
        valid_items = []
        for item in items:
            mfu_str = item.get('MFU', '')
            if mfu_str != '':
                try:
                    float(mfu_str)
                    valid_items.append(item)
                except (ValueError, TypeError):
                    pass

        if not valid_items:
            continue

        sorted_items = sorted(valid_items, key=lambda x: float(x['MFU']))
        min_row = dict(sorted_items[0])
        max_row = dict(sorted_items[-1])
        representative_row = sorted_items[0]

        mfu_values = [float(item['MFU']) for item in sorted_items]
        avg_mfu = _round(sum(mfu_values) / len(mfu_values))
        mid_mfu = _round(statistics.median(mfu_values))

        total_duration = _round(sum(float(item.get('Duration(us)', 0)) for item in items))
        op_count = len(items)
        time_ratio = _round((total_duration / model_runtime) if model_runtime > 0 else 0)
        phase_counts = defaultdict(int)
        for item in items:
            phase_counts[item.get(RECOMPUTE_STAGE_COL, PHASE_FORWARD)] += 1

        contribution_value = _round(avg_mfu * time_ratio)
        if any(item.get(RECOMPUTE_STAGE_COL) == PHASE_RECOMPUTE for item in items):
            group_stage = PHASE_RECOMPUTE
        elif any(item.get(RECOMPUTE_STAGE_COL) == PHASE_BACKWARD for item in items):
            group_stage = PHASE_BACKWARD
        else:
            group_stage = PHASE_FORWARD

        max_row[new_col_name] = 'max'
        # max_row[RECOMPUTE_STAGE_COL] = group_stage
        # max_row['op_count'] = op_count
        # max_row['total_duration(us)'] = total_duration
        # max_row['model_runtime(us)'] = model_runtime
        # max_row['time_ratio(%)'] = _round(time_ratio)

        min_row[new_col_name] = 'min'
        # min_row[RECOMPUTE_STAGE_COL] = group_stage
        # min_row['op_count'] = op_count
        # min_row['total_duration(us)'] = total_duration
        # min_row['model_runtime(us)'] = model_runtime
        # min_row['time_ratio(%)'] = _round(time_ratio)

        avg_row = create_empty_row(new_headers)
        avg_row[new_col_name] = 'ave'
        avg_row['MFU'] = avg_mfu
        avg_row[RECOMPUTE_STAGE_COL] = group_stage
        avg_row['op_count'] = op_count
        avg_row['total_duration(us)'] = total_duration
        avg_row['model_runtime(us)'] = model_runtime
        avg_row['time_ratio(%)'] = _round(time_ratio)
        if group_stage == PHASE_RECOMPUTE:
            avg_row['contribution_to_model_hfu'] = contribution_value
        else:
            avg_row['contribution_to_model_mfu'] = contribution_value

        group_records.append({
            'key': key,
            'rows': [max_row, min_row, avg_row],
            'representative': representative_row,
            'stage': group_stage,
            'forward_signature': get_forward_signature(representative_row),
            'io_key': get_matmul_io_key(representative_row),
            'phase_counts': dict(phase_counts),
            'op_count': op_count,
            'first_start': get_first_start_time(items),
        })

    ordered_group_records = order_statistics_group_records(group_records)
    complete_io_keys = {
        io_key for io_key, relation in get_matmul_relationships(group_records).items()
        if relation.get('is_complete')
    }

    is_first_group = True
    previous_group_record = None
    for group_record in ordered_group_records:
        if is_first_group:
            empty_rows = [create_empty_row(new_headers) for _ in range(10)]
            new_rows.extend(empty_rows)
            is_first_group = False
        else:
            same_complete_matmul = (
                previous_group_record is not None and
                group_record.get('io_key') == previous_group_record.get('io_key') and
                is_complete_matmul_stat_record(previous_group_record, complete_io_keys) and
                is_complete_matmul_stat_record(group_record, complete_io_keys)
            )
            if not same_complete_matmul:
                empty_rows = [create_empty_row(new_headers) for _ in range(2)]
                new_rows.extend(empty_rows)
        new_rows.extend(group_record['rows'])
        previous_group_record = group_record

    for col in ['op_count', 'total_duration(us)', 'model_runtime(us)', 'time_ratio(%)']:
        if col not in new_headers:
            if 'MFU' in new_headers:
                mfu_idx = new_headers.index('MFU')
                new_headers = new_headers[:mfu_idx+1] + [col] + new_headers[mfu_idx+1:]
            else:
                new_headers.append(col)

    if RECOMPUTE_STAGE_COL not in new_headers:
        if 'time_ratio(%)' in new_headers:
            tr_idx = new_headers.index('time_ratio(%)')
            new_headers = new_headers[:tr_idx+1] + [RECOMPUTE_STAGE_COL] + new_headers[tr_idx+1:]
        else:
            new_headers.append(RECOMPUTE_STAGE_COL)

    contribution_cols = ['contribution_to_model_mfu', 'contribution_to_model_hfu']
    missing_contribution_cols = [col for col in contribution_cols if col not in new_headers]
    if missing_contribution_cols:
        stage_idx = new_headers.index(RECOMPUTE_STAGE_COL)
        new_headers = new_headers[:stage_idx+1] + missing_contribution_cols + new_headers[stage_idx+1:]

    stat_output_cols = [
        'op_count',
        'total_duration(us)',
        'model_runtime(us)',
        'time_ratio(%)',
        RECOMPUTE_STAGE_COL,
        'contribution_to_model_mfu',
        'contribution_to_model_hfu',
    ]
    new_headers = [h for h in new_headers if h not in stat_output_cols]
    if 'MFU' in new_headers:
        mfu_idx = new_headers.index('MFU')
        new_headers = new_headers[:mfu_idx+1] + stat_output_cols + new_headers[mfu_idx+1:]
    else:
        new_headers.extend(stat_output_cols)

    return new_rows, new_headers


def main():
    import sys

    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path(r'd:\workfiles\profiles\npu_profiling_ep8cp2_93x480p')

    output_file = base_dir / 'extracted_operators.csv'

    print(f"Searching for _ascend_pt folders in: {base_dir}")

    csv_files = find_ascend_pt_folders(base_dir)
    print(f"Found {len(csv_files)} kernel_details.csv files")

    model_runtime = get_model_runtime(base_dir)
    print(f"Model runtime (min Stage): {model_runtime:.2f} us")

    all_results = []

    for csv_file in csv_files:
        print(f"Processing: {csv_file.parent.parent.name}")
        results = process_kernel_details(csv_file)
        print(f"  Found {len(results)} matching operators")
        all_results.extend(results)

    if not all_results:
        print("No matching operators found!")
        return

    original_headers = list(all_results[0].keys())

    matmul_count = sum(1 for r in all_results if r.get('Type', '').startswith('MatMulV') or
                      (r.get('Type', '').startswith('MatMul') and not r.get('Type', '').startswith('GroupedMatmul')))
    grouped_count = sum(1 for r in all_results if r.get('Type', '').startswith('GroupedMatmul'))
    fa_count = sum(1 for r in all_results if r.get('Type', '') == 'FlashAttentionScore')
    fa_grad_count = sum(1 for r in all_results if r.get('Type', '') == 'FlashAttentionScoreGrad')

    stat_rows, new_headers = compute_statistics(all_results, original_headers, model_runtime)

    all_rows = all_results + stat_rows
    output_rows = []
    for row in all_rows:
        output_row = dict(row)
        if output_row.get('stat_type') != 'ave':
            output_row[RECOMPUTE_STAGE_COL] = ''
        output_rows.append(output_row)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nResults written to: {output_file}")
    print(f"Total records: {len(all_results)}")
    print(f"Statistical rows: {len(stat_rows)}")
    print(f"Total rows in output: {len(all_rows)}")
    print(f"\nOperator breakdown:")
    print(f"  MatMul*: {matmul_count}")
    print(f"  GroupedMatmul*: {grouped_count}")
    print(f"  FlashAttentionScore: {fa_count}")
    print(f"  FlashAttentionScoreGrad: {fa_grad_count}")

    matmul_with_mkn = sum(1 for r in all_results if r.get('M', '') != '' and r.get('Type', '').startswith('MatMul'))
    fa_with_bsnd = sum(1 for r in all_results if r.get('B', '') != '' and r.get('Type', '').startswith('FlashAttention'))
    backward_count_warnings = validate_matmul_backward_counts(all_results)

    print(f"\nDetailed breakdown:")
    print(f"  MatMul with M, K, N: {matmul_with_mkn}")
    print(f"  FA with B, N, S, D: {fa_with_bsnd}")
    print(f"\nMM/GMM backward count check:")
    if backward_count_warnings:
        print(f"  WARNING: {len(backward_count_warnings)} shape(s) do not match one logical forward to two backward ops")
        for warning in backward_count_warnings[:20]:
            print(f"    {warning}")
        if len(backward_count_warnings) > 20:
            print(f"    ... {len(backward_count_warnings) - 20} more")
    else:
        print(f"  OK: every MM/GMM with backward matches one logical forward to two backward ops")


if __name__ == '__main__':
    main()
