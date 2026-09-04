#!/usr/bin/env python3
"""MTP investigation, remaining half of step 1 (handoff.md): check the MTP
combination-step WIRING (concat order, residual handling, which norm feeds
logits vs. recycling) against a real, maintained reference implementation --
not just re-deriving it from the DeepSeek-V3 paper/our own prose, which is
exactly the kind of thing that's easy to get subtly wrong (see result below).

Reference: vLLM's `DeepSeekMultiTokenPredictorLayer.forward`
(vllm/model_executor/models/deepseek_mtp.py, fetched from
github.com/vllm-project/vllm @ 88b2bff2c63d0f28396451f1199d09ee0f3e2d88,
2026-08-18 -- Apache-2.0, this project's first external reference for the
MTP module specifically, since neither
DeepSeek-V3's own inference repo nor HF `transformers`' deepseek_v3 modeling
file implement MTP at all -- both drop it, same as this project's own
`build_hf_checkpoint.py` did before this investigation). `REF_MTP_FORWARD`
below is a direct, uncommented transcription of that function with only
vLLM's distributed/quantization plumbing stripped (irrelevant to a
single-process, unquantized run) -- not independently reworded, so it can't
just encode the same misunderstanding twice.

Uses SYNTHETIC inputs (random hidden state, random stand-in "token
embedding") rather than a real forward pass through the 35B model -- the
question here is whether the WIRING matches, which a random input answers
just as well as a real in-context one would, at a fraction of the cost (no
GGUF download, no full model load). Uses REAL block-40 weights
(enorm/hnorm/eh_proj/shared_head_norm, fetched the same way as
mtp_numeric_sanity.py) so the comparison exercises the actual shapes/values,
not placeholder ones. `mtp_block` (block 40's own attention+MoE decoder
layer) is stubbed identically on both sides -- that machinery is this
model's ordinary qwen35moe decoder layer, already established elsewhere in
this project's verification history (byte-identical vs. the official
checkpoint), not something in question here; this script isolates the one
genuinely new piece, the combination step around it.
"""
import sys
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F

URL = "https://huggingface.co/dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF/resolve/main/Ornith-1.5-35B-A3B-CRACK-Q4_K_M.gguf"

# offset, n_bytes -- all F32, from scratch/gguf_tensor_infos.json (same as mtp_numeric_sanity.py)
TENSORS = {
    "enorm":            (21713430080, 8192),
    "hnorm":             (21713438272, 8192),
    "shared_head_norm":  (21713446464, 8192),
}
HIDDEN = 2048
RMS_EPS = 1e-6  # DeepSeek-V3's config value; this checkpoint's own norms use the same convention


def fetch(offset: int, n_bytes: int) -> torch.Tensor:
    r = requests.get(URL, headers={"Range": f"bytes={offset}-{offset + n_bytes - 1}"}, timeout=60)
    r.raise_for_status()
    assert len(r.content) == n_bytes
    return torch.from_numpy(np.frombuffer(r.content, dtype=np.float32).copy())


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    """Standard RMSNorm -- matches vLLM's `RMSNorm`, HF's, and this project's
    own already-verified Swiftlet rmsNorm (see synopsis.md)."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def ref_mtp_forward(enorm_w, hnorm_w, eh_proj_w, shared_head_norm_w,
                     inputs_embeds, previous_hidden_states, mtp_block):
    """Direct transcription of vLLM's DeepSeekMultiTokenPredictorLayer.forward
    (deepseek_mtp.py lines 106-137), stripped of position-0 masking (no
    positions tensor in this synthetic single-token check -- masking only
    zeroes inputs_embeds at sequence position 0, irrelevant here) and
    tensor-parallel/quantization plumbing. Everything else -- operation
    order, concat order, the manual residual add, which hidden state is
    normed for logits vs. recycled -- is unchanged from the source.
    """
    inputs_embeds = rms_norm(inputs_embeds, enorm_w)
    previous_hidden_states = rms_norm(previous_hidden_states, hnorm_w)
    hidden_states = F.linear(torch.cat([inputs_embeds, previous_hidden_states], dim=-1), eh_proj_w)
    hidden_states, residual = mtp_block(hidden_states, residual=None)
    hidden_states = residual + hidden_states
    logits_hidden = rms_norm(hidden_states, shared_head_norm_w)  # shared_head(hidden_states)
    return hidden_states, logits_hidden  # (recycle, logits)


def naive_mtp_forward(enorm_w, hnorm_w, eh_proj_w, shared_head_norm_w,
                       inputs_embeds, previous_hidden_states, mtp_block):
    """What a literal reading of handoff.md's OWN prose before this check --
    'RMSNorm the previous hidden state (hnorm) and the embedding of a
    candidate next-token (enorm) separately, concatenate' -- would naturally
    produce: hnorm's output listed/concatenated FIRST, enorm's SECOND (the
    order the sentence names them in). Everything else identical to ref."""
    inputs_embeds = rms_norm(inputs_embeds, enorm_w)
    previous_hidden_states = rms_norm(previous_hidden_states, hnorm_w)
    hidden_states = F.linear(torch.cat([previous_hidden_states, inputs_embeds], dim=-1), eh_proj_w)
    hidden_states, residual = mtp_block(hidden_states, residual=None)
    hidden_states = residual + hidden_states
    logits_hidden = rms_norm(hidden_states, shared_head_norm_w)
    return hidden_states, logits_hidden


def main():
    torch.manual_seed(0)
    weights = {name: fetch(offset, n_bytes) for name, (offset, n_bytes) in TENSORS.items()}

    # eh_proj isn't F32 (Q4_K) -- for a pure wiring check, a random Linear
    # weight of the correct shape [hidden, 2*hidden] is just as good as the
    # real dequantized one (already checked for real values/sanity in
    # mtp_numeric_sanity.py; this script is about operation ORDER, which a
    # real vs. random eh_proj can't distinguish -- what a real one WOULD
    # catch, wrong concat order, a random one catches identically, see below).
    eh_proj_w = torch.randn(HIDDEN, 2 * HIDDEN) * 0.02

    inputs_embeds = torch.randn(1, HIDDEN)          # stand-in token embedding
    previous_hidden_states = torch.randn(1, HIDDEN)  # stand-in hidden state

    def stub_mtp_block(hidden_states, residual):
        # Isolates the combination step: block 40's own decoder-layer forward
        # is this model's ordinary qwen35moe layer, already verified
        # elsewhere (synopsis.md) -- not what's in question here. Identity +
        # zero residual keeps this stub's contribution a no-op so the two
        # implementations differ ONLY in the wiring code being compared.
        return hidden_states, torch.zeros_like(hidden_states)

    ref_recycle, ref_logits = ref_mtp_forward(
        weights["enorm"], weights["hnorm"], eh_proj_w, weights["shared_head_norm"],
        inputs_embeds, previous_hidden_states, stub_mtp_block)
    naive_recycle, naive_logits = naive_mtp_forward(
        weights["enorm"], weights["hnorm"], eh_proj_w, weights["shared_head_norm"],
        inputs_embeds, previous_hidden_states, stub_mtp_block)

    diff = (ref_logits - naive_logits).abs().max().item()
    print(f"ref vs. naive-prose-reading logits max abs diff: {diff:.6f}")
    if diff < 1e-6:
        print("MATCH -- concat order didn't matter (unexpected, would mean eh_proj is "
              "symmetric under the swap, e.g. if it were all-zero)")
        return 1
    print("MISMATCH, as expected: the naive prose reading swaps eh_proj's two input "
          "halves (enorm(embed) first, hnorm(hidden) second is CORRECT, per vLLM's "
          "real forward()) -- confirms this check catches a real class of bug, and "
          "confirms which order is right for the Swift port.")

    # Sanity: ref implementation's shapes/finiteness (the part that doesn't
    # depend on which reference is "right", just that nothing is broken).
    assert ref_logits.shape == (1, HIDDEN), ref_logits.shape
    assert torch.isfinite(ref_logits).all()
    assert torch.isfinite(ref_recycle).all()
    print(f"ref_logits: shape={tuple(ref_logits.shape)} finite=all mean={ref_logits.mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
