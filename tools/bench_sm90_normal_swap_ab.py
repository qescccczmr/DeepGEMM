#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import subprocess
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(1, str(REPO_ROOT / "tests"))

import deep_gemm  # noqa: E402
from deep_gemm.testing import bench_kineto, calc_diff, get_arch_major  # noqa: E402
from generators import KernelType, MajorTypeAB, QuantConfig, generate_normal  # noqa: E402


DEFAULT_SHAPES = (
    (1, 4096, 1024),
    (1, 4096, 4096),
    (1, 8192, 4096),
    (8, 4096, 1024),
    (8, 4096, 4096),
    (16, 8192, 256),
    (32, 4096, 256),
)

SWAP_ENV_KEYS = (
    "DG_SM90_SWAP_AB_NORMAL",
    "DG_SM90_SWAP_AB_BLOCK_M",
    "DG_SM90_SWAP_AB_BLOCK_N",
    "DG_SM90_SWAP_AB_CLUSTER_M",
    "DG_SM90_SWAP_AB_CLUSTER_N",
)


def parse_shapes(raw_shapes: str):
    if not raw_shapes:
        return DEFAULT_SHAPES

    shapes = []
    for raw_shape in raw_shapes.split(","):
        fields = raw_shape.lower().replace("x", " ").split()
        if len(fields) != 3:
            raise ValueError(f"invalid shape: {raw_shape!r}, expected MxNxK")
        shapes.append(tuple(int(field) for field in fields))
    return shapes


def percentile(sorted_values, percentile):
    index = max(0, min(len(sorted_values) - 1, math.ceil(len(sorted_values) * percentile) - 1))
    return sorted_values[index]


def set_swap_env(enabled: bool, args):
    old_env = {key: os.environ.get(key) for key in SWAP_ENV_KEYS}
    for key in SWAP_ENV_KEYS:
        os.environ.pop(key, None)

    if enabled:
        os.environ["DG_SM90_SWAP_AB_NORMAL"] = "1"
        if args.swap_block_m is not None:
            os.environ["DG_SM90_SWAP_AB_BLOCK_M"] = str(args.swap_block_m)
        if args.swap_block_n is not None:
            os.environ["DG_SM90_SWAP_AB_BLOCK_N"] = str(args.swap_block_n)
        if args.swap_cluster_m is not None:
            os.environ["DG_SM90_SWAP_AB_CLUSTER_M"] = str(args.swap_cluster_m)
        if args.swap_cluster_n is not None:
            os.environ["DG_SM90_SWAP_AB_CLUSTER_N"] = str(args.swap_cluster_n)

    return old_env


def restore_env(old_env):
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def bench_one(m, n, k, quant_config, recipes, args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    a, b, c, d, ref_d = generate_normal(
        m,
        n,
        k,
        MajorTypeAB.KMajor,
        MajorTypeAB.KMajor,
        False,
        torch.bfloat16,
        KernelType.Kernel1D2D,
        use_ue8m0=False,
        quant_config=quant_config,
    )

    recipe, recipe_a, recipe_b = recipes
    def run():
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

    run()
    diff = calc_diff(d, ref_d).item()
    if args.check and diff >= quant_config.max_diff():
        raise RuntimeError(f"diff too large for m={m}, n={n}, k={k}: {diff}")

    timings = sorted(
        bench_kineto(run, "gemm_", suppress_kineto_output=True) * 1e6
        for _ in range(args.repeat)
    )
    return {
        "min_us": timings[0],
        "median_us": statistics.median(timings),
        "p90_us": percentile(timings, 0.90),
        "diff": diff,
    }


def child_cache_dir(base_cache_dir, mode, shape):
    if not base_cache_dir:
        return None
    m, n, k = shape
    return str(Path(base_cache_dir) / mode / f"m{m}_n{n}_k{k}")


def run_child(shape, use_swap, args):
    m, n, k = shape
    mode = "swap" if use_swap else "default"
    env = os.environ.copy()
    for key in SWAP_ENV_KEYS:
        env.pop(key, None)
    if use_swap:
        env["DG_SM90_SWAP_AB_NORMAL"] = "1"
        if args.swap_block_m is not None:
            env["DG_SM90_SWAP_AB_BLOCK_M"] = str(args.swap_block_m)
        if args.swap_block_n is not None:
            env["DG_SM90_SWAP_AB_BLOCK_N"] = str(args.swap_block_n)
        if args.swap_cluster_m is not None:
            env["DG_SM90_SWAP_AB_CLUSTER_M"] = str(args.swap_cluster_m)
        if args.swap_cluster_n is not None:
            env["DG_SM90_SWAP_AB_CLUSTER_N"] = str(args.swap_cluster_n)

    isolated_cache_dir = child_cache_dir(os.environ.get("DG_JIT_CACHE_DIR"), mode, shape)
    if isolated_cache_dir is not None:
        env["DG_JIT_CACHE_DIR"] = isolated_cache_dir

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "run",
        "--shapes",
        f"{m}x{n}x{k}",
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
    ]
    if not args.check:
        command.append("--no-check")

    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stdout, file=sys.stderr)
        raise RuntimeError(f"{mode} child failed for m={m}, n={n}, k={k}")

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line.removeprefix("RESULT_JSON="))

    print(completed.stdout, file=sys.stderr)
    raise RuntimeError(f"{mode} child did not emit RESULT_JSON for m={m}, n={n}, k={k}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("compare", "run"), default="compare", help=argparse.SUPPRESS)
    parser.add_argument("--shapes", default="", help="comma-separated shapes, e.g. 1x4096x1024,1x4096x4096")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--swap-block-m", type=int)
    parser.add_argument("--swap-block-n", type=int)
    parser.add_argument("--swap-cluster-m", type=int)
    parser.add_argument("--swap-cluster-n", type=int)
    args = parser.parse_args()

    if get_arch_major() != 9:
        raise RuntimeError("this benchmark is intended for SM90 GPUs")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    shapes = parse_shapes(args.shapes)
    quant_config = QuantConfig()
    recipes = quant_config.get_recipes()

    if args.mode == "run":
        if len(shapes) != 1:
            raise ValueError("--mode run expects exactly one shape")
        result = bench_one(*shapes[0], quant_config, recipes, args)
        print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
        return

    print(
        "m,n,k,"
        "default_min_us,swap_min_us,speedup_min,"
        "default_median_us,swap_median_us,speedup_median,"
        "default_p90_us,swap_p90_us,speedup_p90,"
        "diff"
    )
    for m, n, k in shapes:
        default = run_child((m, n, k), False, args)
        swap = run_child((m, n, k), True, args)
        print(
            f"{m},{n},{k},"
            f"{default['min_us']:.3f},{swap['min_us']:.3f},{default['min_us'] / swap['min_us']:.4f},"
            f"{default['median_us']:.3f},{swap['median_us']:.3f},{default['median_us'] / swap['median_us']:.4f},"
            f"{default['p90_us']:.3f},{swap['p90_us']:.3f},{default['p90_us'] / swap['p90_us']:.4f},"
            f"{swap['diff']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
