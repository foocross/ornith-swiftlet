#!/usr/bin/env python3
"""Dequantize the Ornith-1.5-35B-A3B-CRACK-Q4_K_M GGUF and re-emit it as a
plain HF-format safetensors checkpoint dir that `mlx_lm.convert` can load and
requantize to MLX affine INT4/group-64 (the format Swiftlet's `swiftlet-repack
--source` / QwenConfig.swift already expect, verified directly against the
Swiftlet source in ../Swiftlet/Sources/SwiftletCore/).

Text-only, K=8 baseline: drops blk.40 (the MTP "nextn" block) and any vision
tensors (none present in this GGUF; vision lives in the separate mmproj file),
matching the scope ornith-swiftlet-port's overlay already assumes.

Set `ORNITH_KEEP_MTP_BLOCK=1` to keep blk.40 instead of dropping it (see
MTP_PASSTHROUGH below and handoff.md "MTP investigation"). Off by default so
the existing text-only pipeline's output is byte-for-byte unaffected. This
flag has only been exercised against real block-40 tensor bytes at the
mapping/shape level (`scratch/test_mtp_block_mapping.py`, no GGUF download
needed) -- an actual full pipeline run with it on, and mlx_lm.convert's
handling of the resulting extra `model.layers.40.*` keys, is UNVERIFIED.

Tensor naming: everything is emitted WITHOUT any "language_model."/"model."
wrapper prefix (flat `model.embed_tokens...`, `model.layers.N...`,
`lm_head.weight`). mlx_lm's Model.sanitize() fallback branch
(`else: key = "language_model." + key`) adds the correct prefix uniformly --
verified against mlx-lm's actual qwen3_5_moe.py / qwen3_5.py source.

Routed experts are emitted directly under the FINAL mlx-native
`mlp.switch_mlp.{gate,up,down}_proj.weight` names (GGUF already stores them
gate/up/down-separate, not fused, so there's nothing to round-trip through
sanitize's gate_up_proj-split path).

conv1d is emitted already in mlx-native (channels, kernel, 1) layout so
sanitize's has_unsanitized_conv1d auto-detection reads False -- verified
against real tensor values that norm weights are already final-form
(mean ~1.0), NOT delta-from-1, so the auto "+1.0" correction must NOT fire.
"""
import gc
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors.torch import save_file

GGUF_PATH = Path("downloads/Ornith-1.5-35B-A3B-CRACK-Q4_K_M.gguf")
OUT_DIR = Path("hf-checkpoint")
BASE_CONFIG_DIR = Path("ref")  # config.json / tokenizer files pulled from ornith-ai/Ornith-1.5-35B-A3B
SHARD_TARGET_BYTES = 4 * 1024**3  # ~4GB per shard, keeps peak RAM bounded on an 18GB machine

MTP_BLOCK = 40  # confirmed via GGUFReader: blk.40.nextn.* tensors == the MTP draft layer
KEEP_MTP_BLOCK = os.environ.get("ORNITH_KEEP_MTP_BLOCK") == "1"  # opt-in, see module docstring

# --- GatedDeltaNet value-head reorder ---------------------------------------
# llama.cpp's GGUF export lays out the 32 value-heads as [16 "primary" heads,
# then 16 "secondary" heads] (a split/tile grouping of the 16 key-heads x 2
# repeat factor), while mlx-lm's GatedDeltaNet expects them consecutive per
# key-head: [kh0_v0, kh0_v1, kh1_v0, kh1_v1, ...]. Verified against the real
# official checkpoint: sorted(A_log) matched to 5.6e-8, only the per-element
# order differed, and the exact permutation is
# np.arange(32).reshape(16, 2).T.flatten() -- i.e. gguf.reshape(2,16).T.
# Applies to every tensor whose rows/cols are organized by the 32 value-heads:
# A_log, dt_bias (scalar per head), in_proj_a/in_proj_b (row-per-head), and
# in_proj_z / the v-slice of in_proj_qkv / out_proj (row- or col-GROUPS of
# head_v_dim=128 per head). ssm_norm (128,) has no head axis and needs no fix
# (confirmed identical to the official checkpoint already).
NUM_K_HEADS = 16
NUM_V_HEADS = 32
V_PER_K = NUM_V_HEADS // NUM_K_HEADS
HEAD_V_DIM = 128
KEY_DIM = 2048


def regroup_1d(arr: np.ndarray) -> np.ndarray:
    assert arr.shape == (NUM_V_HEADS,), arr.shape
    return arr.reshape(V_PER_K, NUM_K_HEADS).transpose(1, 0).reshape(NUM_V_HEADS).copy()


def regroup_rows(arr: np.ndarray) -> np.ndarray:
    assert arr.shape[0] == NUM_V_HEADS, arr.shape
    hidden = arr.shape[1]
    return arr.reshape(V_PER_K, NUM_K_HEADS, hidden).transpose(1, 0, 2).reshape(NUM_V_HEADS, hidden).copy()


def regroup_row_groups(arr: np.ndarray) -> np.ndarray:
    assert arr.shape[0] == NUM_V_HEADS * HEAD_V_DIM, arr.shape
    hidden = arr.shape[1]
    return (arr.reshape(V_PER_K, NUM_K_HEADS, HEAD_V_DIM, hidden)
               .transpose(1, 0, 2, 3).reshape(NUM_V_HEADS * HEAD_V_DIM, hidden).copy())


def regroup_col_groups(arr: np.ndarray) -> np.ndarray:
    assert arr.shape[1] == NUM_V_HEADS * HEAD_V_DIM, arr.shape
    hidden = arr.shape[0]
    return (arr.reshape(hidden, V_PER_K, NUM_K_HEADS, HEAD_V_DIM)
               .transpose(0, 2, 1, 3).reshape(hidden, NUM_V_HEADS * HEAD_V_DIM).copy())


def fix_in_proj_qkv(arr: np.ndarray) -> np.ndarray:
    # rows: [0:2048)=q, [2048:4096)=k (both 16-head, no reorder needed --
    # the head-count mismatch that causes the reorder is specific to the
    # 16->32 value-head expansion), [4096:8192)=v (needs the row-group fix).
    q_part, k_part, v_part = arr[:KEY_DIM], arr[KEY_DIM:2 * KEY_DIM], arr[2 * KEY_DIM:]
    return np.concatenate([q_part, k_part, regroup_row_groups(v_part)], axis=0)


def fix_conv1d(arr: np.ndarray) -> np.ndarray:
    # conv1d is depthwise over the SAME concatenated [q(2048),k(2048),v(4096)]
    # channel ordering that in_proj_qkv produces (it's literally applied to
    # in_proj_qkv's output), so its row layout needs the identical fix:
    # q/k rows untouched, v rows (last 4096, 32 head-groups of 128) reordered.
    # arr is 3D here: (conv_dim, kernel, 1); regroup_row_groups only cares
    # about axis 0 vs a flattened remainder, so reshape/restore around it.
    q_part, k_part, v_part = arr[:KEY_DIM], arr[KEY_DIM:2 * KEY_DIM], arr[2 * KEY_DIM:]
    kernel = arr.shape[1]
    v_fixed = regroup_row_groups(v_part.reshape(v_part.shape[0], -1)).reshape(v_part.shape[0], kernel, 1)
    return np.concatenate([q_part, k_part, v_fixed], axis=0)

# --- name mapping -----------------------------------------------------------
# All targets below are RELATIVE (no "model."/"language_model." prefix);
# mlx_lm's sanitize() fallback prepends "language_model." to every key that
# doesn't already start with "language_model." or "model.language_model",
# which for our flat names always means "prepend it" -- verified against the
# real source, not assumed.

PASSTHROUGH = {
    # gguf name suffix (after "blk.N.")            -> target suffix (after "model.layers.N.")
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    # full-attention layers
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    # linear-attention (GatedDeltaNet) layers
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    # MoE router / shared expert
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",  # needs reshape (see below)
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    # routed experts -- final mlx-native switch_mlp naming directly
    "ffn_gate_exps.weight": "mlp.switch_mlp.gate_proj.weight",
    "ffn_up_exps.weight": "mlp.switch_mlp.up_proj.weight",
    "ffn_down_exps.weight": "mlp.switch_mlp.down_proj.weight",
}
# handled specially (shape transform needed): ssm_conv1d.weight, ffn_gate_inp_shexp.weight

# blk.40's own attention+MoE tensors need no entry here: they're the same
# full-attention-MoE-layer family as blk.39 (verified: every shared suffix's
# dims/ggml_type matches blk.39's exactly, scratch/gguf_tensor_infos.json),
# so they resolve through PASSTHROUGH above unchanged. Only the four
# MTP-specific combination tensors need new names. Naming follows
# DeepSeek-V3's own published HF checkpoint convention (the architecture
# match this project already established, see handoff.md "MTP investigation"):
# the MTP module's combination step keeps the SAME "model.layers.{N}."
# prefix as an ordinary decoder layer, not a separate top-level module.
MTP_PASSTHROUGH = {
    "nextn.eh_proj.weight": "eh_proj.weight",
    "nextn.enorm.weight": "enorm.weight",
    "nextn.hnorm.weight": "hnorm.weight",
    "nextn.shared_head_norm.weight": "shared_head.norm.weight",
}

TOP_LEVEL = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def target_key(gguf_name: str, keep_mtp: bool = KEEP_MTP_BLOCK) -> str | None:
    """Resolve a GGUF tensor name to its output safetensors key, or None to
    skip it (only ever blk.40 when `keep_mtp` is False). Raises ValueError
    for anything genuinely unmapped -- fail loud, don't silently drop, same
    discipline as every other tensor-naming decision in this project."""
    m = re.match(r"^blk\.(\d+)\.(.+)$", gguf_name)
    if not m:
        if gguf_name not in TOP_LEVEL:
            raise ValueError(f"unmapped top-level tensor: {gguf_name}")
        return TOP_LEVEL[gguf_name]
    layer, suffix = int(m.group(1)), m.group(2)
    if layer == MTP_BLOCK and not keep_mtp:
        return None
    if suffix == "ssm_conv1d.weight":
        return f"model.layers.{layer}.linear_attn.conv1d.weight"
    if suffix in PASSTHROUGH:
        return f"model.layers.{layer}.{PASSTHROUGH[suffix]}"
    if layer == MTP_BLOCK and suffix in MTP_PASSTHROUGH:
        return f"model.layers.{layer}.{MTP_PASSTHROUGH[suffix]}"
    raise ValueError(f"unmapped tensor: {gguf_name}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    reader = GGUFReader(str(GGUF_PATH))
    print(f"opened {GGUF_PATH}: {len(reader.tensors)} tensors", flush=True)

    weights_index: dict[str, str] = {}
    shard_num = 0
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    total_written = 0
    skipped_mtp = 0

    def flush_shard():
        nonlocal shard, shard_bytes, shard_num
        if not shard:
            return
        shard_num += 1
        name = f"model-{shard_num:05d}.safetensors"
        save_file(shard, str(OUT_DIR / name), metadata={"format": "pt"})
        for k in shard:
            weights_index[k] = name
        print(f"  wrote {name}: {len(shard)} tensors, {shard_bytes/1e9:.2f} GB", flush=True)
        shard = {}
        shard_bytes = 0
        gc.collect()

    tensors = sorted(reader.tensors, key=lambda t: t.name)
    for i, t in enumerate(tensors):
        name = t.name
        m = re.match(r"^blk\.(\d+)\.(.+)$", name)

        out_key = target_key(name, keep_mtp=KEEP_MTP_BLOCK)
        if out_key is None:
            skipped_mtp += 1
            continue
        suffix = m.group(2) if m else None

        arr = dequantize(t.data, t.tensor_type).astype(np.float32)

        if m and suffix == "ssm_conv1d.weight":
            # gguf logical shape (conv_dim, kernel_size) -> mlx-native
            # (conv_dim, kernel_size, 1); last-dim==1 so sanitize's
            # has_unsanitized_conv1d check reads False (no spurious +1.0).
            assert arr.ndim == 2, arr.shape
            arr = arr[:, :, None]
            arr = fix_conv1d(arr)
        elif m and suffix == "ffn_gate_inp_shexp.weight":
            # gguf collapsed the trivial out_dim=1 axis -> restore nn.Linear(dim,1) shape
            assert arr.ndim == 1, arr.shape
            arr = arr.reshape(1, -1)
        elif m and suffix == "ssm_a":
            # gguf stores -exp(A_log) (ggml's SSM convention), not A_log itself,
            # AND uses the split (not consecutive) value-head grouping.
            assert arr.ndim == 1 and (arr < 0).all(), (name, arr.shape)
            arr = np.log(-arr)
            arr = regroup_1d(arr)
        elif m and suffix == "ssm_dt.bias":
            arr = regroup_1d(arr)
        elif m and suffix in ("ssm_alpha.weight", "ssm_beta.weight"):  # in_proj_a / in_proj_b
            arr = regroup_rows(arr)
        elif m and suffix == "attn_gate.weight":  # in_proj_z
            arr = regroup_row_groups(arr)
        elif m and suffix == "attn_qkv.weight":  # in_proj_qkv (v-slice only)
            arr = fix_in_proj_qkv(arr)
        elif m and suffix == "ssm_out.weight":  # out_proj (input/column side)
            arr = regroup_col_groups(arr)

        tt = torch.from_numpy(arr)
        if tt.dtype == torch.float32:
            tt = tt.to(torch.bfloat16)
        tt = tt.contiguous()

        shard[out_key] = tt
        shard_bytes += tt.numel() * tt.element_size()
        total_written += 1

        if shard_bytes >= SHARD_TARGET_BYTES:
            flush_shard()

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(tensors)} gguf tensors", flush=True)

    flush_shard()
    print(f"done: {total_written} tensors written, {skipped_mtp} MTP tensors skipped", flush=True)

    index = {
        "metadata": {"total_size": sum((OUT_DIR / f).stat().st_size for f in set(weights_index.values()))},
        "weight_map": weights_index,
    }
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "vocab.json", "merges.txt",
                  "added_tokens.json", "chat_template.jinja"]:
        src = BASE_CONFIG_DIR / fname
        if src.exists():
            (OUT_DIR / fname).write_bytes(src.read_bytes())
            print(f"  copied {fname}", flush=True)

    print(f"wrote checkpoint to {OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
