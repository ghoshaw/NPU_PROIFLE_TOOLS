# Interactive Profile Parser

`extract_profile_interactive.py` parses Ascend profile folders and writes operator, statistics, vector MBU, and communication CSV files. It is the recommended entrypoint when FlashAttention shapes need manual FLOPs parameters.

## Usage

Run from this repository directory:

```powershell
python -B .\extract_profile_interactive.py <profile_path> --device-flops 432 --num 1
```

Arguments:

- `<profile_path>` can be a single `*_pt` / `*_ascend_pt` profile folder, or a parent folder that contains multiple profile folders.
- `--device-flops` is required. Use the device peak compute value in TFLOPS, for example `432` for A5 or `354` for A3.
- `--num` defaults to `1`. It controls how many profile folders participate in operator/statistics/vector extraction when `<profile_path>` is a parent folder.
- Communication extraction scans all `*_pt` folders under the given parent folder and is not limited by `--num`.

Example:

```powershell
python -B .\extract_profile_interactive.py D:\workfiles\profiles\npu_profiling_ep8cp2_93x480p\g340-cd51-4900-b3a3-dd70-c632-a29f_126420_20251106133407217_ascend_pt --device-flops 432
```

## Outputs

The script writes CSV files next to the input path:

- `interactive_extracted_operators.csv`: extracted MM/GMM/FA/FAG operator rows with inferred shapes, FLOPs, MFU, runtime ratio, and MFU/HFU contribution.
- `interactive_operator_statistics.csv`: shape-level statistics for extracted MM/GMM/FA/FAG operators.
- `interactive_vector_operators.csv`: AI vector core operators with logical bytes, estimated bytes, MBU, and byte estimation rule.
- `interactive_comm_operators.csv`: communication events extracted from `trace_view.json`.

## FlashAttention Input

When the script sees a new FA/FAG shape, it prompts:

```text
Enter B,N,Sq,Sk,D,SN,causal:
```

The FLOPs formula is:

```text
FA FLOPs = 4 * B * Sq * Sk * D * N * SN / causal
FAG FLOPs = 2.5 * FA FLOPs
```

Rules:

- Enter seven positive values separated by commas or spaces.
- `B` is batch size.
- `N` is attention head count.
- `Sq` is query sequence length.
- `Sk` is key/value sequence length.
- `D` is head dimension.
- `SN` is the sequence group count used by the model shape expansion.
- `causal` is the divisor for causal masking. Use `1` for full attention and usually `2` for triangular causal attention.
- A matching `FlashAttentionScoreGrad` reuses the same FA parameters and will not prompt again for the same attention shape.

### BNSD self-attention

If Q/K/V are all shaped `[1,12,38160,128]`, use:

```text
1,12,38160,38160,128,1,1
```

For triangular causal attention with the same shape, use:

```text
1,12,38160,38160,128,1,2
```

### BNSD cross-attention

If Q is `[1,24,19080,128]` and K/V are `[1,24,512,128]`, use:

```text
1,24,19080,512,128,1,1
```

### TND format

For Q/K/V shaped `[T,N,D]`, take `N` and `D` directly from the shape. Fill `B`, `Sq`, `Sk`, and `SN` from the real token layout used by the model.

Usually:

```text
Tq = B * Sq * SN
Tk = B * Sk * SN
```

Example: if Q is `[38160,12,128]`, K/V are `[38160,12,128]`, and the real layout is `B=1`, `Sq=38160`, `Sk=38160`, `SN=1`, use:

```text
1,12,38160,38160,128,1,1
```
