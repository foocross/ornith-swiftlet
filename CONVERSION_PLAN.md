# Running Ornith-1.5-35B-A3B-CRACK-Q4_K_M.gguf on Swiftlet

## Decision

Chosen path (Option 1 from the initial scoping): dequantize the GGUF, rebuild a
raw HF-format safetensors checkpoint, requantize with `mlx_lm` to the exact
MLX affine INT4/group-64 format Swiftlet's qpack repacker expects, then run
`swiftlet-repack --source` and the existing `ornith-swiftlet-port` overlay
completely unchanged. No new Swift code.

```
Ornith-1.5-35B-A3B-CRACK-Q4_K_M.gguf   (llama.cpp K-quant, arch tag "qwen35moe")
        | dequantize (gguf python pkg) + rename tensors to raw HF convention
        v
local HF-format checkpoint dir (bf16/fp32 safetensors + config.json from
ornith-ai/Ornith-1.5-35B-A3B, text_config nested — same architecture, CRACK
only changes weight values)
        | python -m mlx_lm.convert -q --q-bits 4 --q-group-size 64
        v
local MLX-lm runtime-format checkpoint (switch_mlp.{gate,up,down}_proj
.weight/.scales/.biases, shape[0] == num_experts) -- same shape the official
ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit repo has
        | swiftlet-repack --source <dir> --output ornith-1.5-35b-crack.qpack
        v
.qpack container
        | apply swiftlet-overlay/patches/000{1,2}-*.patch, swift build -c release
        v
.build/release/swiftlet generate ornith-1.5-35b-crack.qpack --gpu --chat --cache-gb 2
```

## Why this works without touching Swiftlet's Swift source

`Sources/SwiftletRepack/main.swift` + `Sources/SwiftletCore/Qpack.swift` in the
Swiftlet checkout confirm `swiftlet-repack --source <local-dir>` accepts any
local directory in **mlx-lm's own post-quantization runtime layout** — it
looks for `model.layers.{n}.mlp.switch_mlp.{gate_proj,up_proj,down_proj}
.{weight,scales,biases}` with `shape[0] == numExperts`, and errors explicitly
if it instead finds raw HF `mlp.experts.*` tensors ("convert it with mlx-lm
first"). `QwenConfig.swift` parses the exact nested `text_config` config.json
shape already used by `ornith-ai/Ornith-1.5-35B-A3B`, including the
`language_model.` weight-prefix auto-detection
(`Checkpoint.swift:74-77`). So the only new work is producing a correct
mlx-lm-runtime-format checkpoint from the GGUF weights — swiftlet-repack and
the ornith-swiftlet-port overlay need nothing new.

## GGUF -> HF tensor-name mapping (verified against `gguf` package's
`tensor_mapping.py` / `constants.py`, arch `qwen35moe`, and mlx-lm's
`qwen3_5.py` / `qwen3_next.py` source directly — not guessed)

| Concept | GGUF tensor | Raw HF key (pre-mlx_lm-sanitize) |
|---|---|---|
| Token embedding | `token_embd.weight` | `model.language_model.embed_tokens.weight` |
| Final norm | `output_norm.weight` | `model.language_model.norm.weight` |
| LM head | `output.weight` | `lm_head.weight` |
| Per-layer input norm | `blk.N.attn_norm.weight` | `model.language_model.layers.N.input_layernorm.weight` |
| Per-layer post-attn norm | `blk.N.attn_post_norm.weight` | `model.language_model.layers.N.post_attention_layernorm.weight` |
| Full-attn Q/K/V/O (layers where `(N+1)%4==0`) | `blk.N.attn_{q,k,v,output}.weight` | `...self_attn.{q,k,v,o}_proj.weight` |
| Full-attn Q/K norm | `blk.N.attn_{q,k}_norm.weight` | `...self_attn.{q,k}_norm.weight` |
| DeltaNet fused QKV (linear layers) | `blk.N.attn_qkv.weight` | `...linear_attn.in_proj_qkv.weight` |
| DeltaNet gate | `blk.N.attn_gate.weight` | `...linear_attn.in_proj_z.weight` |
| DeltaNet alpha/beta | `blk.N.ssm_alpha.weight` / `blk.N.ssm_beta.weight` | `...linear_attn.in_proj_a.weight` / `...linear_attn.in_proj_b.weight` |
| DeltaNet conv1d | `blk.N.ssm_conv1d.weight` | `...linear_attn.conv1d.weight` (native PyTorch layout, **not** pre-transposed -- see note) |
| DeltaNet A_log / dt_bias | `blk.N.ssm_a.weight` / (dt_bias -- confirm presence in file) | `...linear_attn.A_log` / `...linear_attn.dt_bias` |
| DeltaNet gated norm | `blk.N.ssm_norm.weight` | `...linear_attn.norm.weight` |
| DeltaNet out proj | `blk.N.ssm_out.weight` | `...linear_attn.out_proj.weight` |
| Router | `blk.N.ffn_gate_inp.weight` | `...mlp.gate.weight` |
| Shared-expert gate | `blk.N.ffn_gate_inp_shexp.weight` | `...mlp.shared_expert_gate.weight` |
| Shared expert | `blk.N.ffn_{gate,up,down}_shexp.weight` | `...mlp.shared_expert.{gate,up,down}_proj.weight` |
| Routed experts | **either** `blk.N.ffn_gate_up_exps.weight` (fused) **or** separate `blk.N.ffn_{gate,up}_exps.weight` -- confirm against the real file | `...mlp.experts.gate_up_proj` (fused, split by mlx_lm's own `sanitize()`) -- if GGUF has them split, concatenate gate+up back into one fused tensor along axis -2 before writing, so mlx_lm's stock sanitize path (which expects the fused form) applies unmodified |
| Routed expert down | `blk.N.ffn_down_exps.weight` | `...mlp.experts.down_proj` |

## Important non-obvious correctness note (do not "fix" this)

`TextModel.sanitize()` in mlx-lm's `qwen3_5.py` auto-detects "legacy" raw
PyTorch tensor layout by checking whether `conv1d.weight`'s last dim is `!=
1`. A genuine, never-before-converted HF/PyTorch checkpoint (and therefore
also GGUF, which was exported from that same checkpoint) *always* satisfies
that check, because PyTorch's native `Conv1d` weight shape is `(out_channels,
in_channels/groups, kernel_size)` -- kernel_size last, not 1. When that flag
fires, mlx_lm automatically (a) transposes conv1d into mlx's native layout,
**and** (b) adds `+1.0` to `input_layernorm` / `post_attention_layernorm` /
`model.norm` / `q_norm` / `k_norm` weights (but *not* the DeltaNet's own
`linear_attn.norm`). This is the real, intended storage convention for this
architecture (delta-from-1 residual-style RMSNorm init), not a legacy-only
edge case.

**Action:** keep the dequantized conv1d weight in native (out, in/groups,
kernel) layout and do *not* pre-add 1.0 to any norm weight yourself -- let
mlx_lm's stock `sanitize()` do both, exactly as it would for a real
from-scratch conversion. Reshape only what's needed to match PyTorch's 3D
conv1d convention if GGUF stores it collapsed to 2D (needs confirming against
the real tensor shapes).

## Status: DONE. Working end to end on real Metal hardware.

`scratch/ornith-1.5-35b-crack.qpack` (18GB) runs correctly with
`.build/release/swiftlet generate ... --gpu --chat --cache-gb 2`, producing
coherent, on-topic, correctly-formatted reasoning output (verified with two
separate prompts, one 80 tokens and one 200 tokens, both fully coherent).
~4.9-7.8 tok/s on an M3 Pro with a 2GB expert-cache budget, ~55-60% cache hit
rate (baseline; see "Decode throughput" below for the current number).
`scratch/build_hf_checkpoint.py` is the reusable GGUF -> HF converter;
rerun it against a fresh `scratch/downloads/*.gguf`, then:

```sh
python3 -m mlx_lm convert --hf-path hf-checkpoint --mlx-path mlx-checkpoint \
  -q --q-bits 4 --q-group-size 64 --q-mode affine --dtype bfloat16
swiftlet-repack --source mlx-checkpoint --output out.qpack
```

### Two real bugs found and fixed, both verified against the official
`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` checkpoint's actual dequantized
values (not guessed):

1. **`ssm_a` is `-exp(A_log)`, not `A_log`.** GGML's SSM convention stores
   the ready-to-use decay term, not the log-parameterization mlx-lm's
   `GatedDeltaNet` expects. Fix: `A_log = log(-ssm_a)`.
2. **GatedDeltaNet's 32 value-heads use a different grouping convention
   between llama.cpp and mlx-lm.** With 16 key-heads and a 2x value-head
   expansion, llama.cpp's GGUF export lays the 32 heads out *split*
   (`[16 primary heads][16 secondary heads]`), while mlx-lm expects them
   *consecutive per key-head* (`[kh0_v0,kh0_v1, kh1_v0,kh1_v1, ...]`).
   Verified exactly: `sorted(A_log)` matched the official checkpoint to
   float32 precision (5.6e-8), only the per-element order differed, and the
   exact permutation is `np.arange(32).reshape(16,2).T.flatten()`. This
   permutation had to be applied to **every** tensor organized by value-head:
   `A_log`, `dt_bias` (1 scalar/head), `in_proj_a`/`in_proj_b` (1 row/head),
   and `in_proj_z` / the v-slice of `in_proj_qkv` / `out_proj` / **the
   v-slice of `conv1d`** (groups of 128 rows/cols per head) -- conv1d was
   missed on the first pass (its q/k portions matched exactly, only the
   v-portion was wrong, caught by comparing all three slices separately
   rather than trusting a whole-tensor diff).

First attempt (permutation fix only applied to `A_log`/`dt_bias`/the
`in_proj_*` matrices, not `conv1d`) built and ran but produced token salad --
proof that "the pipeline runs" is not proof of correctness. What actually
caught both bugs: comparing dequantized tensor values elementwise against the
real official checkpoint (`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`, same
architecture, different weights) rather than trusting shape/statistics checks
alone. A 35B-parameter model with subtly wrong linear-attention state can
still load, run, and generate tokens at a normal rate with a perfectly
plausible expert-cache hit rate -- it just generates garbage.

## Disk/RAM plan (tight: 120GB free, 18GB RAM) -- worked, needs care

Do the dequant tensor-by-tensor (never materialize the whole model in RAM).
Sequence: keep the 21.7GB GGUF only until the ~65-70GB bf16 HF dir is fully
written, delete the GGUF, run mlx_lm quantize (~18GB output), delete the
bf16 dir, then repack (~18GB qpack). Hit "No space left on device" once when
an old checkpoint from a prior run wasn't cleaned up first -- always `rm -rf`
the previous run's `mlx-checkpoint`/`*.qpack` before restarting, don't just
trust the free-space math.

## Decode throughput: measured, then fixed the serial-fill bottleneck

`design/BENCHMARK_PLAN.md` (Gate 4) flags "visible read stalls and idle GPU"
as the trigger for prototyping bounded parallel reads instead of assuming it.
Measured it instead of assuming it: added a `fillSeconds`/`maxFillSeconds`
counter to `ExpertCache` isolating time actually spent inside `pread` on a
cache miss, separate from `QwenMetalModel`'s existing GPU
`waitUntilCompleted` timing. Same qpack, same `--cache-gb 2 --chat` prompt as
the original benchmark (reproduced its exact 28,093-miss/60%-hit-rate
baseline):

```text
decode: 200 tokens in 34.5s (5.79 tok/s)
  GPU wait (incl. real GPU exec 12.5s): 16.8s (49%)
  expert-cache fill (pread), fully serial, GPU idle throughout: 15.7s (46%)
  everything else (routing/norms/RoPE/attention/encoding): ~2.0s (6%)
```

Serial fill was nearly as expensive as all GPU work combined, and 100% dead
time (`ExpertCache.buffers()` calls `readExpert` synchronously between the
router GEMV readback and the expert GEMVs, so nothing is dispatched to the
GPU while it runs). Fix: bounded concurrent `pread` within one layer's
top-K miss batch (`Swiftlet/Sources/SwiftletCore/ExpertCache.swift`), guarded
by making `QpackExpertReader`'s fd-open thread-safe
(`Swiftlet/Sources/SwiftletCore/Qpack.swift`) since `pread`'s explicit offset
already makes concurrent reads on one fd safe. Width swept via
`SWIFTLET_EXPERT_READ_CONCURRENCY` (default 8, no rebuild needed to change
it):

| Width | decode | tok/s | fill time |
|---|---|---|---|
| 1 (serial baseline) | 35.1s | 5.69 | 16.24s |
| 2 | 25.0s | 8.00 | 8.67s |
| 4 | 22.7s | 8.81 | 6.95s |
| 8 (default) | 22.5s | 8.89 | 6.93s |
| 16 | 22.8s | 8.79 | 7.10s |

Gains saturate at width 4-8 (matches the batch size: at most `numExpertsPerTok`
misses to fill per layer, so wider than that buys nothing here). Generated
token IDs verified byte-identical to the width-1 baseline at every width --
this changes fill scheduling only, not model output. Net: **~56% decode
throughput improvement (5.79 -> 8.89 tok/s) with zero behavior change**, on
top of the existing overlay/pipeline, at this cache budget and prompt length.

### `--cache-gb` sweep: not a lever here

Tried raising the cache budget (2/4/6/8 GB) expecting a further free win.
Hit rate climbed a lot (60% -> 90%, misses 28,093 -> 7,060) but decode wall
barely moved (22.9s -> 21.9s, within noise) -- GPU-side dispatch/scheduling
overhead grew alongside the larger resident-buffer set (wait-minus-real-exec
went from 2.76s to 4.83s), largely offsetting the fill-time savings. Kept
the default at `--cache-gb 2`; this isn't a lever worth reaching for on this
workload.

### The fast-path fill/GPU-dispatch overlap that wasn't

Reasoned (before reading the actual hot path) that fill and GPU dispatch
could be pipelined per-expert for a further ~45%. Wrong: the real decode
path is `QwenMetalModel.stepOneFast`/`encodePendingMoE`, not the
`moeForward` function read first. It already defers layer N's MoE compute
into layer N+1's command buffer -- a real pipelining optimization already
in place. That closes off the overlap: layer N's expert picks come from a
router GEMV inside layer N's own command buffer (so fill can't start before
that buffer's `waitUntilCompleted` returns), and layer N+1's first op reads
`hBuf`, which only gets a correct value from layer N's `weighted_accum` (a
true data dependency, not just scheduling). The one independent piece --
the shared expert's GEMV chain, which doesn't touch cache buffers -- is
only ~1/8 the routed-experts' GPU cost per layer (`shared_expert_intermediate_size`
== `moe_intermediate_size` == 512, `num_experts_per_tok` == 8), so
overlapping just that would hide an estimated 1-2% of decode wall. Not
worth the correctness risk of restructuring a barrier-synchronized,
register-offset Metal kernel pipeline for that return. Left alone.

### Measurement bug found and fixed, then a real 6% found and fixed

The expert-cache stats line was cumulative for the process's whole
lifetime, so the "fill" figures above for a `--chat` run actually included
prefill's contribution, not just decode's. Fixed by snapshotting the cache
counters right after prefill; corrected decode-only fill was ~5.1s of the
above runs, not 7.1s (prefill alone: ~2,990 misses, ~1.2s). With that
corrected, ~13% of decode wall was still unaccounted for. Ruled out the
CLI's per-step argmax-over-vocab scan and full token-list re-decode
(measured: 0.05s combined, negligible). Found a real cost inside
`ExpertCache` itself: `slotForFill`'s LFU-eviction victim search was an
O(slots) linear scan over every allocated slot (up to 1,213 at
`--cache-gb 2`) on every miss once the cache filled -- 1.3s (6% of decode
wall) over 25k+ decode misses. Fixed with a min-heap over `(freq, lastUse)`
and lazy staleness checking on pop (a fresh candidate is pushed on every
touch; a pop that no longer matches the slot's current freq/lastUse is
just discarded, no per-touch removal bookkeeping needed), compacted
periodically so a long-lived server process doesn't grow it unboundedly.
Pure CPU/data-structure change, no Metal involved, and correctness doesn't
depend on which slot an eviction picks (only hit rate/speed does) -- lower
risk than the fast-path idea above by construction, not just in practice.
Measured: bookkeeping 1.30s -> 0.12s (91% reduction); decode 21.6s -> 20.0s
(9.27 -> 9.98 tok/s); output byte-identical across runs.

**Running total: 5.79 -> 9.98 tok/s (~72% cumulative decode throughput
improvement), `--cache-gb 2`, this qpack/prompt.** Not yet swept across
other cache budgets or longer runs (`design/BENCHMARK_PLAN.md` Gate 3/4
still call for that more broadly); ~13% of decode wall remains attributed
only in aggregate (small per-layer CPU costs inside `stepOneFast` --
attention-core softmax, per-layer router softmax, buffer/array copies --
no single dominant piece found on this pass).
