#!/usr/bin/env python3
"""Unit-test `build_hf_checkpoint.py`'s ORNITH_KEEP_MTP_BLOCK=1 path against
REAL blk.40 tensor bytes -- without downloading the full 21.7GB GGUF or
running the full ~65GB checkpoint conversion (see handoff.md "MTP
investigation", step 2). Companion to `scratch/mtp_numeric_sanity.py` (which
checked the nextn.* weights aren't garbage); this checks the NAME MAPPING
and SHAPE/DEQUANT pipeline for the whole of block 40, both new (nextn.*) and
reused (block 40's own attention+MoE tensors, same family as blk.39) parts.

Fetches block 40 -- confirmed contiguous on disk, offset 21166759488 to EOF,
~521MB -- in one HTTP Range GET (no local GGUF needed), then for every
tensor in it: dequantizes exactly as `build_hf_checkpoint.main()` would, and
checks the result against `target_key()`'s mapping plus basic sanity
(shape, no NaN/Inf).
"""
import os
import sys
from pathlib import Path

import numpy as np
import requests
from gguf import GGMLQuantizationType
from gguf.quants import dequantize, quant_shape_to_byte_shape

sys.path.insert(0, str(Path(__file__).parent))
from build_hf_checkpoint import MTP_BLOCK, MTP_PASSTHROUGH, PASSTHROUGH, target_key

URL = "https://huggingface.co/dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF/resolve/main/Ornith-1.5-35B-A3B-CRACK-Q4_K_M.gguf"
BLOCK40_START = 21166759488
FILE_END = 21713462848  # blk.40.post_attention_norm.weight's end == EOF

# name -> (offset, n_bytes, ggml_type, dims-as-stored-in-file/ne-order)
BLOCK40_TENSORS = {
    "blk.40.attn_k.weight":                 (21166759488,    589824, "Q4_K", [2048, 512]),
    "blk.40.attn_k_norm.weight":             (21167349312,      1024, "F32",  [256]),
    "blk.40.attn_norm.weight":               (21167350336,      8192, "F32",  [2048]),
    "blk.40.attn_output.weight":             (21167358528,   4718592, "Q4_K", [4096, 2048]),
    "blk.40.attn_q.weight":                  (21172077120,   9437184, "Q4_K", [2048, 8192]),
    "blk.40.attn_q_norm.weight":             (21181514304,      1024, "F32",  [256]),
    "blk.40.attn_v.weight":                  (21181515328,    860160, "Q6_K", [2048, 512]),
    "blk.40.ffn_down_exps.weight":           (21182375488, 220200960, "Q6_K", [512, 2048, 256]),
    "blk.40.ffn_down_shexp.weight":          (21402576448,    860160, "Q6_K", [512, 2048]),
    "blk.40.ffn_gate_exps.weight":           (21403436608, 150994944, "Q4_K", [2048, 512, 256]),
    "blk.40.ffn_gate_inp.weight":            (21554431552,   2097152, "F32",  [2048, 256]),
    "blk.40.ffn_gate_inp_shexp.weight":      (21556528704,      8192, "F32",  [2048]),
    "blk.40.ffn_gate_shexp.weight":          (21556536896,    589824, "Q4_K", [2048, 512]),
    "blk.40.ffn_up_exps.weight":             (21557126720, 150994944, "Q4_K", [2048, 512, 256]),
    "blk.40.ffn_up_shexp.weight":            (21708121664,    589824, "Q4_K", [2048, 512]),
    "blk.40.nextn.eh_proj.weight":           (21708711488,   4718592, "Q4_K", [4096, 2048]),
    "blk.40.nextn.enorm.weight":             (21713430080,      8192, "F32",  [2048]),
    "blk.40.nextn.hnorm.weight":             (21713438272,      8192, "F32",  [2048]),
    "blk.40.nextn.shared_head_norm.weight":  (21713446464,      8192, "F32",  [2048]),
    "blk.40.post_attention_norm.weight":     (21713454656,      8192, "F32",  [2048]),
}

assert sum(v[1] for v in BLOCK40_TENSORS.values()) == FILE_END - BLOCK40_START, \
    "BLOCK40_TENSORS doesn't exactly tile [BLOCK40_START, FILE_END) -- offsets copy/paste error"


def fetch_block40() -> bytes:
    # Dev-loop cache outside the repo (never committed) -- avoids re-fetching
    # ~521MB on every iteration of this script while writing/debugging it.
    cache = os.environ.get("MTP_BLOCK40_CACHE")
    if cache and Path(cache).exists():
        print(f"using cached {cache}", flush=True)
        return Path(cache).read_bytes()
    print(f"fetching blk.40 ({(FILE_END - BLOCK40_START)/1e6:.1f} MB, one Range GET)...", flush=True)
    r = requests.get(URL, headers={"Range": f"bytes={BLOCK40_START}-{FILE_END - 1}"}, timeout=300)
    r.raise_for_status()
    assert len(r.content) == FILE_END - BLOCK40_START, len(r.content)
    if cache:
        Path(cache).write_bytes(r.content)
    return r.content


def dequant(raw: bytes, ggml_type: str, dims: list[int]) -> np.ndarray:
    if ggml_type == "F32":
        # GGUFReader reshapes F32 tensors to reversed-dims before dequantize()
        # ever sees them (dequantize() is an identity op for float types) --
        # replicate that reshape here since we're building the array from raw
        # bytes directly instead of through GGUFReader.
        return np.frombuffer(raw, dtype=np.float32).reshape(tuple(reversed(dims)))
    qtype = getattr(GGMLQuantizationType, ggml_type)
    np_dims = tuple(reversed(dims))
    byte_shape = quant_shape_to_byte_shape(np_dims, qtype)
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(byte_shape)
    return dequantize(packed, qtype)


def main():
    blob = fetch_block40()

    # --- 1. target_key() mapping checks (no network/dequant needed) --------
    for name in BLOCK40_TENSORS:
        suffix = name.split(".", 2)[2]
        # Default path (keep_mtp=False): every blk.40 tensor is skipped,
        # UNCHANGED from before this feature existed -- the one thing that
        # must never regress.
        assert target_key(name, keep_mtp=False) is None, \
            f"{name}: default path must still skip blk.40 (got a key back)"

        got = target_key(name, keep_mtp=True)
        assert got is not None, f"{name}: keep_mtp=True should map it, got None"
        assert got.startswith(f"model.layers.{MTP_BLOCK}."), got
        if suffix in MTP_PASSTHROUGH:
            assert got == f"model.layers.{MTP_BLOCK}.{MTP_PASSTHROUGH[suffix]}", got
        elif suffix == "ssm_conv1d.weight":
            assert False, f"{name}: block 40 should never have a GatedDeltaNet suffix"
        else:
            assert suffix in PASSTHROUGH, f"{name}: suffix not in PASSTHROUGH or MTP_PASSTHROUGH"
            assert got == f"model.layers.{MTP_BLOCK}.{PASSTHROUGH[suffix]}", got
    print(f"target_key(): {len(BLOCK40_TENSORS)}/{len(BLOCK40_TENSORS)} blk.40 tensors map "
          f"correctly under keep_mtp=True, all skipped under keep_mtp=False (unchanged default)", flush=True)

    # --- 2. dequant + shape/sanity checks on real bytes ---------------------
    for name, (offset, n_bytes, ggml_type, dims) in BLOCK40_TENSORS.items():
        rel = offset - BLOCK40_START
        raw = blob[rel:rel + n_bytes]
        assert len(raw) == n_bytes, (name, len(raw), n_bytes)
        arr = dequant(raw, ggml_type, dims).astype(np.float32)

        expected_shape = tuple(reversed(dims))
        assert arr.shape == expected_shape, (name, arr.shape, expected_shape)
        n_nan, n_inf = int(np.isnan(arr).sum()), int(np.isinf(arr).sum())
        assert n_nan == 0 and n_inf == 0, (name, n_nan, n_inf)

        suffix = name.split(".", 2)[2]
        if suffix.endswith("norm.weight") or suffix == "nextn.shared_head_norm.weight":
            # RMSNorm gains: mean should be near 1.0, not near 0 (the "+1.0
            # correction" false lead build_hf_checkpoint.py's docstring warns
            # about) and not wildly dispersed.
            assert 0.9 < arr.mean() < 1.1, (name, float(arr.mean()))
            assert arr.std() < 0.1, (name, float(arr.std()))
        print(f"  ok: {name}: shape={arr.shape} mean={arr.mean():.4f} std={arr.std():.4f}", flush=True)

    print("all block-40 mapping + dequant checks passed", flush=True)


if __name__ == "__main__":
    sys.exit(main())
