import os
import random

import torch

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major
from generators import MajorTypeAB, QuantConfig, generate_m_grouped_contiguous


def test_sm90_m_grouped_contiguous_swap_ab_native() -> None:
    if get_arch_major() != 9:
        return

    old_value = os.environ.get("DG_SM90_SWAP_AB_NATIVE")
    os.environ["DG_SM90_SWAP_AB_NATIVE"] = "1"
    try:
        quant_config = QuantConfig()
        recipe, recipe_a, recipe_b = quant_config.get_recipes()
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)

        for n in (512, 1024):
            for k in (128, 256):
                torch.manual_seed(0)
                random.seed(0)

                m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(
                    4,
                    128,
                    n,
                    k,
                    MajorTypeAB.KMajor,
                    MajorTypeAB.KMajor,
                    use_ue8m0=False,
                    use_psum_layout=False,
                    quant_config=quant_config,
                )

                deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                    a,
                    b,
                    d,
                    grouped_layout,
                    disable_ue8m0_cast=True,
                    use_psum_layout=False,
                    recipe=recipe,
                    recipe_a=recipe_a,
                    recipe_b=recipe_b,
                )

                diff = calc_diff(d, ref_d)
                assert diff < quant_config.max_diff(), (m, n, k, diff)
    finally:
        if old_value is None:
            os.environ.pop("DG_SM90_SWAP_AB_NATIVE", None)
        else:
            os.environ["DG_SM90_SWAP_AB_NATIVE"] = old_value





