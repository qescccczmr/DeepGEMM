import argparse
import sys

import torch

import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff
from generators import MajorTypeAB, QuantConfig, generate_m_grouped_contiguous


def make_paired_case(num_groups: int, expected_m: int, n: int, k: int):
    quant_config = QuantConfig()
    m_padded, a, b, labels, d_padded, ref_padded = generate_m_grouped_contiguous(
        num_groups, expected_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        quant_config=quant_config,
    )
    labels_cpu = labels.cpu()
    row_ids, compact_layout, start = [], [], 0
    for group_idx in range(num_groups):
        ids = torch.where(labels_cpu == group_idx)[0].cuda()
        row_ids.append(ids)
        compact_layout.append((start, ids.numel()))
        start += ids.numel()
    row_ids = torch.cat(row_ids)
    a_compact = (a[0].index_select(0, row_ids).contiguous(),
                 a[1].index_select(0, row_ids).contiguous())
    d_compact = torch.empty((start, n), dtype=d_padded.dtype, device=d_padded.device)
    ref_compact = ref_padded.index_select(0, row_ids)
    compact_layout = torch.tensor(compact_layout, dtype=torch.int32, device='cuda')
    return quant_config, m_padded, start, a, a_compact, b, labels, compact_layout, d_padded, d_compact, ref_compact


def run_case(num_groups: int, expected_m: int, n: int, k: int, correctness_only: bool):
    (quant_config, m_padded, m_valid, a_padded, a_compact, b, labels,
     compact_layout, d_padded, d_compact, ref_compact) = make_paired_case(
        num_groups, expected_m, n, k)
    recipe, recipe_a, recipe_b = quant_config.get_recipes()
    kwargs = dict(recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
    workspace = torch.empty(deep_gemm.get_num_sms() * 3 * 128,
                            dtype=torch.uint8, device='cuda')

    deep_gemm.m_grouped_fp8_fp4_gemm_nt_compact(
        a_compact, b, d_compact, compact_layout, workspace=workspace, **kwargs)
    torch.cuda.synchronize()
    diff = calc_diff(d_compact, ref_compact)
    if diff >= quant_config.max_diff():
        per_group = [float(calc_diff(d_compact[s:s + size], ref_compact[s:s + size]))
                     for s, size in compact_layout.cpu().tolist()]
        raise AssertionError((float(diff), per_group, compact_layout.cpu().tolist()))
    if correctness_only:
        print(f'correct: G={num_groups}, valid/padded={m_valid}/{m_padded}, diff={diff:.5f}')
        return

    def run_padded():
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a_padded, b, d_padded, labels, **kwargs)

    def run_compact():
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_compact(
            a_compact, b, d_compact, compact_layout, workspace=workspace, **kwargs)

    padded_s = bench_kineto(run_padded, 'gemm_', suppress_kineto_output=True)
    compact_s = bench_kineto(run_compact, 'gemm_', suppress_kineto_output=True)
    print(f'G={num_groups:2}, expected_m={expected_m:4}, N={n:5}, K={k:5}, '
          f'valid/padded={m_valid}/{m_padded} ({m_valid / m_padded:.1%}), '
          f'padded={padded_s * 1e6:.1f} us, compact={compact_s * 1e6:.1f} us, '
          f'speedup={padded_s / compact_s:.3f}x')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    deep_gemm.set_mk_alignment_for_contiguous_layout(
        deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())
    cases = ((4, 97, 256, 512),) if args.correctness_only else (
        # Compact-stress cases (the existing contiguous suite uses much larger M).
        (32, 20, 4096, 2048),
        (32, 64, 4096, 2048),
        (16, 256, 4096, 4096),
        # Shapes from test_m_grouped_gemm_contiguous.
        (4, 8192, 6144, 7168),
        (4, 8192, 7168, 3072),
        (4, 8192, 4096, 4096),
        (4, 8192, 4096, 2048),
        (8, 4096, 6144, 7168),
        (8, 4096, 7168, 3072),
        (8, 4096, 4096, 4096),
        (8, 4096, 4096, 2048),
    )
    for case in cases:
        run_case(*case, args.correctness_only)


if __name__ == '__main__':
    main()
