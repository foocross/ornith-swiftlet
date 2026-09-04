#!/usr/bin/env python3
"""Prototype INT4/INT8 affine quantization error against REAL cached K/V
values (not synthetic data), per the project's own established discipline
of verifying against real values -- see synopsis.md's tensor-bug hunt.

Input: scratch/kv_dump/kv_layer<N>_<k|v>.f32, produced by running
`SWIFTLET_DUMP_KV=./kv_dump swiftlet generate ...` (see main.swift's
diagnostic dump, gated behind that env var, off by default).

Layout: flat float32, row-major [pos][kvHead][headDim], kvHead=2, headDim=256
-> row stride 512 floats (matches QwenCPUModel.DecodeState's documented
layout and QwenMetalModel.attnCoreCPU's cblas_sgemv lda usage).

Tests the plan's recommended scope (per-token, per-group affine
quantization, matching the existing MLX-affine convention already used for
weights: w[i] = scale[g]*q[i] + bias[g]) at INT4 and INT8, group sizes
32/64/128/256, against the alternative of per-channel-across-time grouping
(the research doc's original proposal) to check whether the added
complexity of per-channel-K actually buys meaningfully better accuracy.
"""
import sys
from pathlib import Path

import numpy as np

ROW_STRIDE = 512  # kvHead(2) * headDim(256)


def load_layer(path: Path) -> np.ndarray:
    flat = np.fromfile(path, dtype="<f4")
    assert flat.size % ROW_STRIDE == 0, f"{path}: size {flat.size} not a multiple of {ROW_STRIDE}"
    return flat.reshape(-1, ROW_STRIDE)  # [pos, 512]


def affine_quant_dequant(x: np.ndarray, bits: int, group_size: int, axis: int) -> np.ndarray:
    """Affine per-group quantize+dequantize along `axis`, groups of
    `group_size` contiguous elements along that axis. Mirrors
    Checkpoint.swift's w[i] = scale[g]*q[i] + bias[g] convention."""
    qmax = (1 << bits) - 1
    x = np.moveaxis(x, axis, -1)
    orig_shape = x.shape
    assert orig_shape[-1] % group_size == 0, f"{orig_shape[-1]} not divisible by group {group_size}"
    groups = x.reshape(*orig_shape[:-1], orig_shape[-1] // group_size, group_size)
    gmin = groups.min(axis=-1, keepdims=True)
    gmax = groups.max(axis=-1, keepdims=True)
    scale = np.where(gmax > gmin, (gmax - gmin) / qmax, 1.0)
    bias = gmin
    q = np.clip(np.round((groups - bias) / scale), 0, qmax)
    deq = scale * q + bias
    deq = deq.reshape(orig_shape)
    return np.moveaxis(deq, -1, axis)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 1.0


def report(label: str, orig: np.ndarray, deq: np.ndarray):
    diff = orig - deq
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    denom = float(np.mean(np.abs(orig))) or 1.0
    rel = mean_abs / denom
    cos = cosine_sim(orig, deq)
    print(f"  {label:38s} max_abs={max_abs:8.5f}  mean_abs={mean_abs:8.5f}  "
          f"rel={rel:7.4f}  cos_sim={cos:.6f}")


def main():
    dump_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "kv_dump")
    layer_files = sorted(dump_dir.glob("kv_layer*_k.f32"))
    if not layer_files:
        print(f"no kv_layer*_k.f32 files in {dump_dir} -- run with "
              f"SWIFTLET_DUMP_KV={dump_dir} first")
        sys.exit(1)

    for kfile in layer_files:
        layer = kfile.stem.replace("kv_layer", "").replace("_k", "")
        vfile = kfile.with_name(kfile.name.replace("_k.f32", "_v.f32"))
        k = load_layer(kfile)
        v = load_layer(vfile)
        print(f"\n=== layer {layer}: K {k.shape}, V {v.shape} ===")
        print(f"  K: mean={k.mean():.4f} std={k.std():.4f} min={k.min():.4f} max={k.max():.4f}")
        print(f"  V: mean={v.mean():.4f} std={v.std():.4f} min={v.min():.4f} max={v.max():.4f}")

        for bits in (8, 4):
            for group in (32, 64, 128, 256):
                # Recommended scope: per-token, group along the row (axis=1,
                # the 512-wide [kvHead*headDim] axis) -- quantize each new
                # token's row independently at append time, no rewriting of
                # earlier tokens ever needed.
                k_deq = affine_quant_dequant(k, bits, group, axis=1)
                v_deq = affine_quant_dequant(v, bits, group, axis=1)
                report(f"INT{bits} g{group} per-token (K)", k, k_deq)
                report(f"INT{bits} g{group} per-token (V)", v, v_deq)

        # Comparison: per-channel-across-time (axis=0, grouping along the
        # growing position axis instead) -- the research doc's original
        # proposal for keys. Only meaningful if there are enough positions;
        # skip degenerate cases.
        print("  -- per-channel-across-time comparison (bits=4) --")
        for group in (32, 64, 128, 256):
            if k.shape[0] < 2 * group:
                continue  # need at least 2 full groups to be meaningful
            usable = (k.shape[0] // group) * group
            k_deq = affine_quant_dequant(k[:usable], 4, group, axis=0)
            v_deq = affine_quant_dequant(v[:usable], 4, group, axis=0)
            report(f"INT4 g{group} per-channel-time (K, n={usable}, {usable // group} groups)", k[:usable], k_deq)
            report(f"INT4 g{group} per-channel-time (V, n={usable}, {usable // group} groups)", v[:usable], v_deq)


if __name__ == "__main__":
    main()
