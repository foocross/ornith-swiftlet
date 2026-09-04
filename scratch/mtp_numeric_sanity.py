#!/usr/bin/env python3
"""Step 1 of the MTP investigation (handoff.md "If pursuing: concrete next
steps"): numeric sanity check on blk.40's nextn.* tensors, without
downloading the full 21.7GB GGUF.

Fetches only the byte ranges we need via HTTP Range requests against the
HF resolve URL (offsets/sizes are absolute file offsets, taken from
scratch/gguf_tensor_infos.json / GGUFReader.data_offset semantics -- verified
contiguous with n_bytes so a single ranged GET covers several tensors at
once).

Checks:
- blk.40.nextn.enorm/hnorm/shared_head_norm.weight (F32, RMSNorm gains):
  do they look like real RMSNorm weights (mean near 1, sane std, no
  NaN/Inf), same shape of check build_hf_checkpoint.py already did for the
  other 40 layers' norms?
- blk.40.attn_norm.weight (a real, already-relied-upon norm from block 40's
  own attention) and blk.40.post_attention_norm.weight, as in-block "known
  good" references for comparison.
- blk.40.nextn.eh_proj.weight (Q4_K, [4096,2048]): dequantize and check
  basic weight-matrix statistics (no NaN/Inf, sane mean/std, not
  degenerate/all-zero).
"""
import sys
import numpy as np
import requests
from gguf import GGMLQuantizationType
from gguf.quants import dequantize, quant_shape_to_byte_shape

URL = "https://huggingface.co/dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF/resolve/main/Ornith-1.5-35B-A3B-CRACK-Q4_K_M.gguf"

# (name, offset, n_bytes, ggml_type, dims-as-stored-in-file/ne-order)
TENSORS = [
    ("blk.40.attn_norm.weight",              21167350336,    8192, "F32",  [2048]),
    ("blk.40.nextn.eh_proj.weight",           21708711488, 4718592, "Q4_K", [4096, 2048]),
    ("blk.40.nextn.enorm.weight",             21713430080,    8192, "F32",  [2048]),
    ("blk.40.nextn.hnorm.weight",             21713438272,    8192, "F32",  [2048]),
    ("blk.40.nextn.shared_head_norm.weight",  21713446464,    8192, "F32",  [2048]),
    ("blk.40.post_attention_norm.weight",     21713454656,    8192, "F32",  [2048]),
]


def fetch(offset: int, n_bytes: int) -> bytes:
    end = offset + n_bytes - 1
    r = requests.get(URL, headers={"Range": f"bytes={offset}-{end}"}, timeout=120)
    r.raise_for_status()
    if len(r.content) != n_bytes:
        raise RuntimeError(f"expected {n_bytes} bytes, got {len(r.content)}")
    return r.content


def stats(name: str, arr: np.ndarray):
    arr = arr.astype(np.float64)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    finite = arr[np.isfinite(arr)]
    print(f"{name}: shape={arr.shape} n={arr.size} "
          f"mean={finite.mean():.4f} std={finite.std():.4f} "
          f"min={finite.min():.4f} max={finite.max():.4f} "
          f"nan={n_nan} inf={n_inf}")


def main():
    for name, offset, n_bytes, ggml_type, dims in TENSORS:
        raw = fetch(offset, n_bytes)
        if ggml_type == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        elif ggml_type == "Q4_K":
            qtype = GGMLQuantizationType.Q4_K
            np_dims = tuple(reversed(dims))
            byte_shape = quant_shape_to_byte_shape(np_dims, qtype)
            packed = np.frombuffer(raw, dtype=np.uint8).reshape(byte_shape)
            arr = dequantize(packed, qtype)
        else:
            raise ValueError(ggml_type)
        stats(name, arr)


if __name__ == "__main__":
    sys.exit(main())
