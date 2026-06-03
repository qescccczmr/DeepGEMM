import os
import random

import torch

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major
from generators import KernelType, MajorTypeAB, QuantConfig, generate_normal


def test_sm90_normal_small_m_swap_ab_native() -> None:
    if get_arch_major() != 9:
        return

    old_value = os.environ.get("DG_SM90_SWAP_AB_NORMAL")
    os.environ["DG_SM90_SWAP_AB_NORMAL"] = "1"
    try:
        quant_config = QuantConfig()
        recipe, recipe_a, recipe_b = quant_config.get_recipes()

        for m in (1, 8, 16, 17, 32, 33, 64, 65, 128):
            for k in (128, 256):
                torch.manual_seed(0)
                random.seed(0)

                a, b, c, d, ref_d = generate_normal(
                    m,
                    512,
                    k,
                    MajorTypeAB.KMajor,
                    MajorTypeAB.KMajor,
                    False,
                    torch.bfloat16,
                    KernelType.Kernel1D2D,
                    use_ue8m0=False,
                    quant_config=quant_config,
                )

                deep_gemm.fp8_fp4_gemm_nt(
                    a,
                    b,
                    d,
                    c=c,
                    disable_ue8m0_cast=True,
                    recipe=recipe,
                    recipe_a=recipe_a,
                    recipe_b=recipe_b,
                )

                diff = calc_diff(d, ref_d)
                assert diff < quant_config.max_diff(), (m, 512, k, diff)
    finally:
        if old_value is None:
            os.environ.pop("DG_SM90_SWAP_AB_NORMAL", None)
        else:
            os.environ["DG_SM90_SWAP_AB_NORMAL"] = old_value
