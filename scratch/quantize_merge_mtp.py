#!/usr/bin/env python3
"""Quantize block 40's (the MTP nextn block) 20 tensors by hand, using the
EXACT same convention mlx_lm.convert applies to the rest of this model, and
merge them into an already-converted mlx-checkpoint/ dir as a new shard.

Bypasses mlx_lm's own Model-class loading entirely for these tensors --
that path is a hard blocker (see handoff.md "MTP investigation": strict
load_weights() rejects any key with no matching module, and there IS no
module for layer 40 in a 40-layer stack; bumping num_hidden_layers to 41
would be actively wrong too, since DecoderLayer's is_linear formula would
then misclassify layer 40 as GatedDeltaNet, not full-attention). Swiftlet's
own MTPGPU loader reads tensors directly by name, never through mlx_lm's
Model class, so this is the correct level to operate at.

Quantization convention reverse-engineered from mlx_lm's OWN real output
(mlx-checkpoint/, produced by the same mlx_lm.convert run against this
model minus block 40) plus its source (qwen3_5.py Qwen3_5TextModel.
quant_predicate, switch_layers.py SwitchLinear.to_quantized):
  - group_size=64, mode="affine" throughout (this project's standard CLI args)
  - bits=8 for paths ending in "mlp.gate" or "shared_expert_gate" (the
    router and the shared-expert gate) -- confirmed by reading the real
    config.json quantization overrides mlx_lm.convert wrote for layers 0-39
  - bits=4 for every other Linear-family tensor (self_attn.{q,k,v,o}_proj,
    mlp.shared_expert.{gate,up,down}_proj, mlp.switch_mlp.{gate,up,down}_proj,
    eh_proj) -- the model's global default
  - RMSNorm vectors (7 of the 20: input_layernorm, post_attention_layernorm,
    self_attn.q_norm/k_norm, enorm, hnorm, shared_head.norm) are never
    quantized -- nn.RMSNorm has no to_quantized, confirmed by the real
    mlx-checkpoint's own norm tensors staying bf16 unquantized.
  - mx.quantize(w, group_size, bits, mode="affine") applied directly to the
    raw weight array (2D for ordinary Linear, 3D for switch_mlp's per-expert
    batch) is exactly what SwitchLinear.to_quantized/nn.Linear.to_quantized
    do internally -- confirmed by reading switch_layers.py source, not
    guessed.
"""
import json
from pathlib import Path

import mlx.core as mx

HF_CKPT = Path("hf-checkpoint")
MLX_CKPT = Path("mlx-checkpoint")
LAYER = 40
PREFIX = f"model.layers.{LAYER}."
TARGET_PREFIX = f"language_model.model.layers.{LAYER}."
GROUP_SIZE = 64
MODE = "affine"
NEW_SHARD = "model-mtp-block40.safetensors"

# suffix (after "model.layers.40.") -> bits, or None if unquantized (kept bf16)
TENSOR_PLAN = {
    "eh_proj.weight": 4,
    "enorm.weight": None,
    "hnorm.weight": None,
    "input_layernorm.weight": None,
    "mlp.gate.weight": 8,
    "mlp.shared_expert.down_proj.weight": 4,
    "mlp.shared_expert.gate_proj.weight": 4,
    "mlp.shared_expert.up_proj.weight": 4,
    "mlp.shared_expert_gate.weight": 8,
    "mlp.switch_mlp.down_proj.weight": 4,
    "mlp.switch_mlp.gate_proj.weight": 4,
    "mlp.switch_mlp.up_proj.weight": 4,
    "post_attention_layernorm.weight": None,
    "self_attn.k_norm.weight": None,
    "self_attn.k_proj.weight": 4,
    "self_attn.o_proj.weight": 4,
    "self_attn.q_norm.weight": None,
    "self_attn.q_proj.weight": 4,
    "self_attn.v_proj.weight": 4,
    "shared_head.norm.weight": None,
}


def target_key(suffix: str) -> str:
    # ".weight" suffix stripped for the base name; quantized ones get
    # .weight/.scales/.biases, unquantized keep the plain .weight name.
    base = suffix[: -len(".weight")]
    return TARGET_PREFIX + base


def main():
    index = json.loads((HF_CKPT / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    src_keys = {k: wm[k] for k in wm if k.startswith(PREFIX)}
    assert set(k[len(PREFIX):] for k in src_keys) == set(TENSOR_PLAN), (
        f"tensor set mismatch: {set(k[len(PREFIX):] for k in src_keys) ^ set(TENSOR_PLAN)}"
    )

    # load only the shard(s) holding layer 40 (mx.load reads a whole shard,
    # but there's exactly one per the earlier inspection).
    shard_files = sorted(set(src_keys.values()))
    raw = {}
    for sf in shard_files:
        raw.update(mx.load(str(HF_CKPT / sf)))

    out_tensors = {}
    bpw_report = []
    for suffix, bits in TENSOR_PLAN.items():
        src_key = PREFIX + suffix
        w = raw[src_key]
        base_target = target_key(suffix)
        if bits is None:
            out_tensors[base_target + ".weight"] = w
            bpw_report.append((suffix, "bf16 (unquantized)"))
            continue
        wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=bits, mode=MODE)
        out_tensors[base_target + ".weight"] = wq
        out_tensors[base_target + ".scales"] = scales
        out_tensors[base_target + ".biases"] = biases
        bpw_report.append((suffix, f"{bits}-bit, group_size={GROUP_SIZE}, shape {list(w.shape)} -> {list(wq.shape)}"))

    print("Quantization plan applied:")
    for suffix, note in bpw_report:
        print(f"  {suffix}: {note}")

    mx.save_safetensors(str(MLX_CKPT / NEW_SHARD), out_tensors, metadata={"format": "mlx"})
    print(f"\nwrote {NEW_SHARD}: {len(out_tensors)} tensors")

    # merge into index.json
    mlx_index_path = MLX_CKPT / "model.safetensors.index.json"
    mlx_index = json.loads(mlx_index_path.read_text())
    for k in out_tensors:
        mlx_index["weight_map"][k] = NEW_SHARD
    mlx_index["metadata"]["total_size"] = sum(
        (MLX_CKPT / f).stat().st_size for f in set(mlx_index["weight_map"].values())
    )
    mlx_index_path.write_text(json.dumps(mlx_index, indent=2))
    print(f"updated index.json: {len(mlx_index['weight_map'])} total tensors")


if __name__ == "__main__":
    main()
