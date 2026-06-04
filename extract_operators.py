import os
import csv
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from collections import defaultdict
import statistics


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
            result_row['FLOPs'] = flops if flops is not None else ''
            result_row['MFU'] = mfu if mfu is not None else ''
            result_row['MBU'] = mbu if mbu is not None else ''
            result_row['I'] = I if I is not None else ''
            result_row['AI'] = 432/4.0
            result_row['source_path'] = csv_path.parent.parent.name if csv_path else ''

            results.append(result_row)

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


def create_empty_row(headers: List[str]) -> Dict[str, str]:
    return {h: '' for h in headers}


def compute_statistics(all_results: List[Dict[str, Any]], headers: List[str], model_runtime: float) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for row in all_results:
        key = get_group_key(row)
        if key:
            groups[key].append(row)

    new_rows = []
    new_col_name = 'stat_type'

    if 'source_path' in headers:
        source_idx = headers.index('source_path')
        new_headers = headers[:source_idx] + [new_col_name] + headers[source_idx:]
    else:
        new_headers = [new_col_name] + headers

    is_first_group = True
    for key, items in sorted(groups.items()):
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

        if is_first_group:
            empty_rows = [create_empty_row(new_headers) for _ in range(10)]
            new_rows.extend(empty_rows)
            is_first_group = False
        else:
            empty_rows = [create_empty_row(new_headers) for _ in range(2)]
            new_rows.extend(empty_rows)

        sorted_items = sorted(valid_items, key=lambda x: float(x['MFU']))
        min_row = dict(sorted_items[0])
        max_row = dict(sorted_items[-1])

        mfu_values = [float(item['MFU']) for item in sorted_items]
        avg_mfu = sum(mfu_values) / len(mfu_values)
        mid_mfu = statistics.median(mfu_values)

        total_duration = sum(float(item.get('Duration(us)', 0)) for item in items)
        op_count = len(items)
        time_ratio = (total_duration / model_runtime) if model_runtime > 0 else 0

        type_val = sorted_items[0].get('Type', '')
        is_fa = type_val.startswith('FlashAttention')
        contribution_mfu = avg_mfu * time_ratio if not is_fa else ''

        max_row[new_col_name] = 'max'
        max_row['op_count'] = op_count
        max_row['total_duration(us)'] = total_duration
        max_row['model_runtime(us)'] = model_runtime
        max_row['time_ratio(%)'] = time_ratio * 100
        new_rows.append(max_row)

        min_row[new_col_name] = 'min'
        min_row['op_count'] = op_count
        min_row['total_duration(us)'] = total_duration
        min_row['model_runtime(us)'] = model_runtime
        min_row['time_ratio(%)'] = time_ratio * 100
        new_rows.append(min_row)

        avg_row = create_empty_row(new_headers)
        avg_row[new_col_name] = 'ave'
        avg_row['MFU'] = avg_mfu
        avg_row['op_count'] = op_count
        avg_row['total_duration(us)'] = total_duration
        avg_row['model_runtime(us)'] = model_runtime
        avg_row['time_ratio(%)'] = time_ratio * 100
        avg_row['contribution_to_model_mfu'] = contribution_mfu
        new_rows.append(avg_row)


    for col in ['op_count', 'total_duration(us)', 'model_runtime(us)', 'time_ratio(%)', 'contribution_to_model_mfu']:
        if col not in new_headers:
            if col == 'contribution_to_model_mfu' and 'time_ratio(%)' in new_headers:
                tr_idx = new_headers.index('time_ratio(%)')
                new_headers = new_headers[:tr_idx+1] + [col] + new_headers[tr_idx+1:]
            elif 'MFU' in new_headers:
                mfu_idx = new_headers.index('MFU')
                new_headers = new_headers[:mfu_idx+1] + [col] + new_headers[mfu_idx+1:]
            else:
                new_headers.append(col)

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

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

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

    print(f"\nDetailed breakdown:")
    print(f"  MatMul with M, K, N: {matmul_with_mkn}")
    print(f"  FA with B, N, S, D: {fa_with_bsnd}")


if __name__ == '__main__':
    main()
