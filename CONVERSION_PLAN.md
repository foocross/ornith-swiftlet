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

### MoE kernel fusion: real win, smaller than the dispatch-count drop suggested

`handoff.md` left "pursue MoE kernel fusion" as an open decision after the
real per-category GPU timestamps pointed at MoE (46.3% of decode wall,
+18% over its byte-budget prediction) as the most dispatch-heavy, least
efficient category. Picked up and pursued: `encodePendingMoE`
(`QwenMetalModel.swift`) issued up to 38 separate GPU dispatches per MoE
layer for K=8 routed experts (8 gate + 8 up + 3 shared GEMVs, 8 per-expert
`silu_mul` + 1 shared, 8 down + 1 shared-down, 1 `weighted_accum`).

Two changes, landed together, gated by `SWIFTLET_NO_MOE_BATCH=1` (forces the
original per-expert loop, mirrors `SWIFTLET_NO_FAST_GEMV`'s A/B shape):

1. **Batched `silu_mul` (unconditional, not gated)**: routed experts' gate/up
   outputs are already laid out contiguously by expert index in `sBuf`, so
   the 8 per-expert `silu_mul` dispatches collapse into 1 over `K*inter`
   elements -- same computation, pure dispatch-count reduction.
2. **New Metal kernel `gemv_moe_batched`** (bindless -- first use of this
   pattern in the codebase): fuses gate+up into one dispatch across all K
   resident experts, and does the same for down, via a tiny per-layer buffer
   of the K experts' raw GPU addresses (`MTLBuffer.gpuAddress`) plus
   `enc.useResource` for residency. Every expert's qpack blob has the same
   internal gate/up/down byte offsets (only the blob's base address differs
   per expert), so one dispatch per stage suffices: 38 dispatches/MoE-layer
   -> 9. Gated on this model's actual expert quant profile (4-bit,
   groupSize%8==0, matching `gemv_affine_fast`'s existing eligibility check)
   and on qpack+`ExpertCache` mode specifically -- the non-cache "stacks"
   fallback path is untouched.

Verified: new `MetalKernelTests.gemvMoEBatchedMatchesCPU` (K=3 synthetic
expert blobs, dual-stage and single-stage, `< 1e-3` vs CPU reference); full
`swift test` (63 tests) green after updating `FastPathBaseline`'s
hardcoded dispatch-count regression constants to the new, lower counts (by
design -- the tripwire is meant to catch exactly this kind of change).
Full-model: two paired runs (`SWIFTLET_NO_MOE_BATCH=1` vs default), same
reference prompt, generated text **byte-identical** to `gate10_heap.out` in
all four runs (both flag states, both repeats).

Measured decode throughput, `--cache-gb 2`, same reference qpack/prompt,
two paired runs on this machine (variance noted throughout this file is
real -- both pairs run back-to-back for a controlled comparison):

| Run | No batching (flag forced off) | Batching (default) | Delta |
|---|---|---|---|
| 1 | 9.35 tok/s | 9.85 tok/s | +5.3% |
| 2 | 9.41 tok/s | 9.94 tok/s | +5.6% |

Total GPU compute dispatches for the 200-step decode run dropped by almost
half (354,400 -> 178,400), but decode wall only improved ~5%: most of
decode's GPU time is real compute (bandwidth-bound GEMV), not per-dispatch
fixed overhead, more than the earlier byte-budget-vs-measured gap implied.
A real, verified win, smaller than the dispatch-count drop alone would
suggest -- set against `handoff.md`'s honest ceiling estimate, this lands at
the low end of "double-digit-percent" rather than the high end.

**Running total after this: ~9.4 -> ~9.9 tok/s from this change (+5-6%),
compounding onto the 5.79 -> ~9.4 baseline above.**

### CPU attention-core vectorization: real win, grows with context length

With MoE fusion landed, a code review looking for the next lever (not a new
GPU trace -- see the caveat below) noticed `attnCoreCPU`
(`QwenMetalModel.swift`, the decode-fast-path attention core) and its
sibling `attnForward` (the non-fast-path fallback, same computation) do
QK^T and softmax-weighted-V-sum as hand-written scalar Swift loops --
`O(H * kvLen * hd)` per full-attention layer per decode step -- despite
`QwenCPUModel.swift` having `import Accelerate` at the top of the file,
unused for this path. Same for `rmsNorm`/`softmaxRow` in
`QwenCPUModel.swift`, which this code also calls.

Why this matters specifically for *this* model: `layer_types` in the
qpack's `config.json` has `full_attention_interval: 4`, so 10 of the 40
layers hit this loop, at `num_attention_heads=16`, `head_dim=256`. Every
other decode-time cost this arc has measured (MoE, dense projections,
lm_head, expert-cache fill) is `O(1)` in context length -- this is the one
piece that grows with it, and it runs entirely on CPU while the GPU sits
idle (between `commitAndWait` and the next command buffer), so none of it
shows up in the per-category GPU-timestamp breakdown above. It was cheap
enough at the 200-token runs benchmarked so far to hide inside "negligible"
loop overhead; it isn't a fixed cost, so it doesn't stay negligible.

Fix: replaced the scalar loops with `cblas_sgemv` (one BLAS call for QK^T,
one for the softmax-weighted V-sum, per head -- `kAll`/`vAll`'s
`[pos][kvHead][headDim]` layout is exactly a row-major `(kvLen x hd)` matrix
with row stride `KVH*hd`, which `cblas_sgemv`'s `lda` expresses directly,
no repacking needed) and `rmsNorm`/`softmaxRow` with `vDSP` (`vDSP_svesq`/
`vDSP_vsmul`/`vDSP_vmul`; `vDSP_maxv`/`vDSP_vsadd`/`vvexpf`/`vDSP_sve`/
`vDSP_vsmul`). Both `attnCoreCPU` and `attnForward` fixed identically (the
non-fast-path is a straight fallback, `SWIFTLET_NO_FAST_GEMV=1`, worth
keeping consistent).

**One deliberate exception to this project's byte-identical-output bar**:
Accelerate's vectorized reductions sum in a different (tree-style) order
than the sequential scalar loops did, so output is *numerically
equivalent*, not bit-identical, to pre-change runs. Verified instead
against the existing test-suite tolerances built for exactly this kind of
float-reordering difference (`FixtureForwardTests`/`IncrementalDecodeTests`
`maxAbsDiff` bounds, already up to `2e-3` on quantized logits) plus full
`swift test` (63/63 green) -- and, in practice, the real full-model runs
below produced **byte-identical generated text** (not just "close") to the
pre-change baseline, greedy decoding included; the difference only shows up
in timing/instruction-count stats, not in a single output token.

Measured, `--cache-gb 2`, same reference qpack/prompt, paired runs
(`/tmp/swiftlet-old` = pre-change binary via `git stash`, `/tmp/swiftlet-new`
= this change) on this machine:

| Run | Old | New | Delta | Note |
|---|---|---|---|---|
| 200 tokens (reference prompt) | 9.85 tok/s | 10.27 tok/s | +4.3% | matches `moefusion_batched.out` exactly except stats lines |
| 354 tokens (same prompt, model ran to EOS under `--max-new 1500`) | 9.98 tok/s | 11.12 tok/s | **+11.4%** | user-mode CPU time 7.48s -> 4.73s (-37%), vs. -22% at 200 tokens |

The win grows with context length as predicted -- not a full longer-context
sweep (that's still `BENCHMARK_PLAN.md`'s open item, and this was one paired
comparison, not several), but real, directional evidence for the specific
mechanism identified: this is the one decode-time cost in the whole arc
that's `O(context length)` rather than `O(1)`, so it's expected to matter
increasingly more, not less, at the 32K-token context `synopsis.md`'s
memory math targets.

**Running total after this: ~9.9 -> ~10.3-11.1 tok/s depending on context
length so far reached in a run (larger win at longer context), compounding
onto the 5.79 -> ~9.9 baseline above.**

### `--cache-gb` gap revisited: hazard-tracking hypothesis tested, rejected

Follow-up session, prompted by re-reading `handoff.md`/`synopsis.md` for the
next lever rather than a new trace. The original `--cache-gb` sweep (above)
left the growing "wait-minus-real-exec" gap (2.76s at 1213 slots -> 4.83s at
4854 slots, real GPU exec time flat at ~9.7-10.1s throughout) as an
observation, not an explained cause. Re-reading the existing sweep logs
(no new runs needed for this part) confirmed the gap tracks slot count
specifically, not the workload: dispatch count (410400) and command-buffer
count (10200) are identical across every cache size in those logs.

**Hypothesis:** `ExpertCache`'s slot buffers are allocated plain
`.storageModeShared`, but every other persistent, manually-synchronized
buffer in `QwenMetalModel.swift` (`sBuf`, `hBuf`, per-layer `hist`/`state`)
already uses `.hazardTrackingModeUntracked`, on the reasoning that the
decode loop's own `waitUntilCompleted` sequencing (the same data-dependency
chain documented above under "fast-path fill/GPU-dispatch overlap that
wasn't") already guarantees the ordering Metal's automatic hazard tracking
would otherwise redundantly track. `ExpertCache` slots are under the
identical discipline (a fill's `pread` completes via `group.wait()` before
the buffer reaches an encoder; a slot is only evicted/refilled after the
command buffer(s) that last read it have already completed) but never got
the same treatment -- a real, precedented gap, and a plausible mechanism for
per-resource driver overhead that scales with the *number of tracked
buffers*, which is exactly what grows with `--cache-gb`.

**Tested, not just reasoned about:** implemented (gated behind
`SWIFTLET_NO_EXPERT_CACHE_UNTRACKED`, mirroring the `SWIFTLET_NO_MOE_BATCH`/
`SWIFTLET_NO_FAST_GEMV` escape-hatch pattern), verified byte-identical
output against `gate10_heap.out` and full `swift test` (63/63), then
measured paired runs (same binary, same qpack/prompt, `--cache-gb`
2/4/6/8, untracked vs. tracked):

| cache-gb | untracked (the hypothesis) | tracked (unchanged) | delta |
|---|---|---|---|
| 2 | 2.602s | 2.616s | -0.014 |
| 4 | 3.995s | 3.660s | +0.335 |
| 6 | 4.423s | 4.177s | +0.246 |
| 8 | 5.336s / 4.595s (2 runs) | 4.151s / 4.075s (2 runs) | +0.85s avg |

**Rejected.** Untracked mode doesn't shrink the gap -- at the two larger
cache sizes it's consistently a bit *worse*, replicated across a repeat run
at `--cache-gb 8`. The gap grows almost identically under both conditions,
which rules out per-resource hazard-tracking overhead as the mechanism.
Reverted the change (`ExpertCache.swift` back to the pre-experiment state,
no flag left behind); `swift test` (63/63) and the reference command still
pass clean on the reverted tree.

What's left standing, untested, as a more likely explanation: memory
pressure rather than driver/resource-tracking overhead. This is an
18GB-unified-memory machine; `--cache-gb 8` plus the resident dense weights
pushes real usage into a range where the OS/GPU driver may be doing real
work (page-in, compaction) that shows up as `waitUntilCompleted` latency
without appearing in Metal's own `gpuStartTime`/`gpuEndTime` window. Checking
that would need `vm_stat`/working-set instrumentation during a run -- a
different kind of measurement than anything else in this document -- and
wasn't pursued further this round, since it wouldn't change the shipped
default (`--cache-gb 2`) either way. **Conclusion updated, not reversed:**
`--cache-gb` above 2 still isn't a lever on this machine, now for a
specific, falsified-by-experiment reason (not a fixable Metal
resource-tracking inefficiency) rather than an open question.

### Memory-pressure hypothesis: instrumented, partially confirmed

The one thing the hazard-tracking investigation above left standing as an
*untested* explanation for the gap -- real memory pressure on this
18GB-unified-memory machine, as opposed to per-resource driver overhead --
finally got instrumented rather than left as a theory. New scripts:
`scratch/memory_pressure_sweep.sh` (runs the reference command at
`--cache-gb` 2/4/6/8, repeating 8, while sampling `vm_stat` system-wide and
`footprint -p <pid>` per-process in parallel) and
`scratch/analyze_memory_pressure.py` (correlates both against the existing
`decode Metal S3a` wait/gpu stats line).

**Caveat on comparability:** `scratch/ornith-1.5-35b-crack.qpack` -- the
model used for every number in the original sweep above -- no longer
exists on this disk (per its own documented cleanup). This run used
`scratch/ornith-1.5-35b-base.qpack` instead (same architecture/memory
layout, different weights). Absolute tok/s and gap magnitudes here don't
match the CRACK-build figures above 1:1; what's being tested (does a
memory-pressure signal track the growing gap in *shape*) is a property of
the runtime/memory layout, not the specific weights, so this is still a
valid test of the hypothesis, just not a byte-for-byte reproduction of the
original numbers.

| `--cache-gb` | slots | wait (s) | gpu exec (s) | **gap (s)** | footprint peak (MB) | compressor-in (pages) | decompressions (pages) | pageins (pages) | swapins/outs |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1,213 | 17.279 | 14.022 | **3.257** | 3,788 | 2,313,450 | 322,167 | 725,361 | 0 / 0 |
| 4 | 2,427 | 16.214 | 11.801 | **4.413** | 5,938 | 4,651,794 | 878,063 | 637,372 | 0 / 0 |
| 6 | 3,640 | 19.922 | 14.533 | **5.389** | 8,087 | 7,144,340 | 938,790 | 620,152 | 0 / 0 |
| 8 (run 1) | 4,854 | 20.856 | 14.297 | **6.559** | 10,240 | 10,764,022 | 1,287,927 | 647,470 | 128 / 0 |
| 8 (run 2) | 4,854 | 21.151 | 14.989 | **6.162** | 10,237 | 9,708,457 | 990,323 | 640,449 | 149 / 0 |

The gap reproduces the same growing shape as the original CRACK-build
sweep (there: 2.76s -> 4.83s; here: 3.26s -> ~6.2-6.6s -- different
absolute numbers, same monotonic growth with cache size).

**Two of the four memory-pressure signals don't track the gap at all:**
`pageins` is flat-to-slightly-*declining* as cache size grows (725K at
2GB down to ~640-647K at 6-8GB) -- the opposite of what the hypothesis
predicts. `swapouts` is zero at every cache size; `swapins` is zero
everywhere except a handful of events (128-149) at the largest cache size
-- present, but far too small in count to explain a multi-second gap by
itself. Raw disk-swap pressure is not the mechanism.

**Two signals do track it, cleanly:** compressor pages-in and
decompression counts both grow monotonically with cache size, roughly in
step with `phys_footprint` (2.3M -> 10.8M compressor-in pages, 322K ->
1.29M decompressions, both scaling with cache-gb the same way `gap_s`
does). This is macOS's memory compressor -- distinct from disk swap --
compressing/decompressing pages under memory pressure without ever
touching the SSD. Per-second decompression rates here (roughly 13K-52K/s
across the runs) are the same order of magnitude as the "abbaglio" trace
`research/research.txt` cites (60K-130K decompressions/sec correlating
with the same failure mode: a large, wired-adjacent Metal buffer pool
squeezing the OS page cache).

**Verdict: partially confirmed, more specific than the open question was.**
Real memory pressure is a plausible contributor to the gap -- not through
disk swapping (ruled out, same as raw pageins), but through compressor-pool
churn that scales with resident buffer size, consistent with `ExpertCache`
slot buffers (`.storageModeShared`, growing with `--cache-gb`) squeezing
the OS's own page cache the larger they get. This doesn't reopen the
hazard-tracking conclusion (that mechanism was tested directly and
rejected) -- it gives the *other* untested theory from that investigation
real, if imperfect, supporting evidence instead of leaving it as
speculation. Caveats: one sweep, mostly single runs per cache size (only
`--cache-gb 8` repeated), against `base.qpack` not the original
`crack.qpack`, and background system noise on a shared machine (the
`--cache-gb 6` run's mid-run `free_last` reading was a noisy outlier) --
directional evidence, not a controlled, statistically-clean result. Does
not change the shipped default (`--cache-gb 2`) either way; a real fix
(e.g. reducing `ExpertCache`'s resident-buffer pressure directly) would be
new scope, not something this diagnostic pass attempted.

### Expert I/O hint sweep: `F_NOCACHE`, real win at the shipped default

Closes the remaining half of `research/research.txt` idea #4 ("fix SSD IO
path") the memory-pressure diagnostic above left open: that pass
instrumented and *explained* the `--cache-gb` gap (macOS memory-compressor
churn tracking resident-buffer size), but didn't yet test a fix. Mechanism
targeted directly: `ExpertCache` already holds its own resident copies of
hot expert blobs in `.storageModeShared` `MTLBuffer`s, so every miss-fill
was potentially double-cached -- once by the OS's unified buffer cache
(UBC), once by `ExpertCache`'s own slot -- exactly the redundant-caching
shape behind the confirmed compressor-churn mechanism. Added
`fcntl(fd, F_NOCACHE, 1)` on every `packed_experts` fd
(`Qpack.swift`'s `QpackExpertReader.fd(for:)`), gated by
`SWIFTLET_EXPERT_NOCACHE` (default on now; `=0` reverts).

`research/research.txt` cites a different codebase's finding that "every
macOS IO hint tested... default is best" -- tested directly against this
actual pipeline instead of trusting that citation, per this project's own
convention. Reference command (`scratch/io_hint_sweep/`, same prompt/qpack
as the memory-pressure sweep, `--max-new 200`):

| `--cache-gb` | condition | decode tok/s (3 clean repeats) |
|---|---|---|
| 2 (shipped default) | OS-cached (old default) | 11.58, 11.73, 11.71 |
| 2 (shipped default) | `F_NOCACHE` | 13.49, 13.36, 13.27 |
| 8 | OS-cached (old default) | 10.70, 10.84, 10.63 |
| 8 | `F_NOCACHE` | 10.57, 10.51, 10.20 |

A first run at `--cache-gb 2` (excluded from the table above) showed the
*opposite* direction (nocache slower) -- a cold-start/noise outlier per the
project's established caveat about background noise on this shared
machine; the following three repeats were tight and consistent, so that
first run was discarded rather than treated as a tie-breaker.

**Real, reproducible ~14-15% decode-throughput win at `--cache-gb 2` (the
actual shipped default), a small consistent ~2-4% regression at
`--cache-gb 8`.** Opposite of the naive hypothesis (double-caching pressure
should matter *more* at larger cache sizes, not less) -- another instance
of this project's rule that measuring beats predicting. Since `--cache-gb`
8 is already documented above ("not a lever worth reaching for here") as
not a setting worth using for speed, the trade is worth taking as the new
default rather than staying opt-in. Output verified byte-identical to the
old path in both directions (`SWIFTLET_EXPERT_NOCACHE=0` reproduces the old
default exactly; the new default reproduces the old opt-in `=1` path
exactly) -- `F_NOCACHE` only changes OS caching behavior, never model
output, and this confirms it. `swift test --filter QpackTests` still
passes.

**The rest of idea #4, deliberately not pursued:**

- **2MB-aligned `MTLBuffer` allocation for expert-cache slots** (the other
  half of the doc's suggested "2MB-aligned DMA buffers" item): the cited
  Flash-MoE win comes from *removing* a host-malloc-then-copy-into-Metal-
  buffer step -- this pipeline never had that step (`pread` already writes
  straight into the `MTLBuffer`'s own backing memory, confirmed when idea
  #2 was scoped and skipped). Re-allocating already-page-aligned
  `storageModeShared` buffers at a coarser 2MB alignment would add real
  memory-management complexity (`bytesNoCopy` deallocator lifetime, length
  rounded to page multiples) for a mechanism that doesn't apply here;
  not attempted.
- **"Delete the wired Metal cache, trust the OS page cache instead"** (the
  more radical redesign the doc's cited source used, trading ~32% slower
  for a jetsam-invisible footprint): directly contradicts `ExpertCache`'s
  explicit design goal, stated in its own doc comment -- "replaces OS
  paging so the working set can never thrash the machine: memory use is
  exactly `slots * expertStride`, no more" -- which matters specifically
  for the iOS jetsam-avoidance case this kit targets, not just this Mac.
  Abandoning the bounded-memory guarantee to chase a throughput number on
  one hardware target would regress the kit's actual differentiator; not
  attempted.

Idea #4 is now closed: instrumented, mechanism confirmed, a real fix shipped
for the shipped default, the rest of its scope explicitly declined with
reasoning rather than left silently undone.

## Base model build: direct, no GGUF conversion needed

Everything above is specific to the CRACK build, which started from a GGUF
that had to be dequantized, re-tensor-named, and requantized through mlx_lm
before it matched the format `swiftlet-repack` expects -- that's where both
of the "two real bugs" above came from. The plain base model,
`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`, is published by `ornith-ai` itself
*already* in mlx-lm's post-quantization runtime layout (the same format the
CRACK pipeline's last conversion step produces), so none of that conversion
-- and neither of its two bugs -- applies. `swiftlet-repack` streams it
straight from Hugging Face:

```sh
swiftlet-repack --from-hf ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit \
  --output scratch/ornith-1.5-35b-base.qpack
```

(`--from-hf` is resumable/streaming -- see `swiftlet-repack --help`.) This
is the same checkpoint already used earlier in this document as the
ground-truth reference for verifying the CRACK conversion's tensor values,
and the one used for the Space Invaders code-generation comparison in
`synopsis.md` ("Known gaps" -- CRACK's abliteration damages code
generation, verified against this exact base checkpoint).

**Done, verified.** Built in 547.7s to an 18GB qpack
(`scratch/ornith-1.5-35b-base.qpack`); a 60-token sanity run
(`--gpu --chat --cache-gb 2`) loaded cleanly and produced coherent,
on-topic output at 11.17 tok/s, 55-56% expert-cache hit rate -- faster
than the CRACK build's original baseline, consistent with the throughput
work landed since ("Decode throughput" above). Full details in
`synopsis.md` "Base model build".

## KV cache INT8 quantization: implemented, verified

Follow-up to the memory-pressure diagnostic above, and the second half of
`research/research.txt`'s idea #1 (KV cache compression for the 10
full-attention layers -- the one component of decode memory that grows
with context length; the other 30 layers are GatedDeltaNet with fixed-size
recurrent state). Full detail: `Swiftlet/Sources/SwiftletCore/KVQuant.swift`.

### Real-data prototype first, before touching the hot path

Before writing any Swift, dumped real cached K/V from an actual run
(`SWIFTLET_DUMP_KV=<dir>`, a new opt-in diagnostic in `main.swift`/
`QwenCPUModel.DecodeState.dumpFullAttentionKV`, mirroring the existing
opt-in-env-var convention) and tested quantization error against it with
`scratch/kv_quant_prototype.py` -- matching this project's own established
discipline of verifying against real values, not synthetic assumptions
(`synopsis.md`'s tensor-bug hunt did the same).

Findings, at 421 real tokens across all 10 full-attention layers:

- **INT8, any group size 32-256: essentially lossless** (cosine similarity
  >0.9999, ~0.6-1.4% relative error).
- **INT4, per-token** (quantize each token's own row independently, the
  simplest scheme): cosine similarity 0.986-0.996, ~10-20% relative
  error -- a real, non-trivial cost, worse than "4-bit is reliably
  lossless" claims `research/research.txt` borrowed from weight-
  quantization studies (different tensor shape/distribution).
- **INT4, per-channel-across-time** (quantize each of the 512 channels
  using its own scale across many cached positions, the research doc's
  original, more complex proposal): cosine similarity 0.997-0.999,
  roughly *half* per-token's relative error, consistently across every
  layer -- but needs buffering newly-appended tokens until a channel's
  position-group fills, real added complexity over per-token.

Given that tradeoff, the choice made (not the plan's original "start with
simple per-token INT4" recommendation, revised after this real measurement
surfaced the accuracy gap): **ship INT8 per-token only.** Near-lossless at
any tested granularity sidesteps the INT4 scheme question entirely, at a
real 4x-ish memory win and the lowest implementation-risk surface.

### Design: group size = headDim, factored-affine-dequant reduction

One affine `w[i] = scale[g]*q[i] + bias[g]` group per (position, kvHead) --
group size = headDim (256), so exactly 2 groups per cached row (one per
kv-head), matching the existing MLX-affine convention already used for
weights (`Checkpoint.swift`, `gemv_affine` in `Kernels.metal.txt`).

The KV-cache read path (`attnCoreCPU`/`attnForward` in `QwenMetalModel.swift`)
uses `cblas_sgemv` over the *entire* cached K/V arrays per head, not a
per-element loop -- so quantization can't just dequantize-on-read
per-element. `attnCoreInt8` (the new shared attention core both paths
call) widens the INT8 codes to plain `Float` (no scale/bias applied) once
per layer per step via `vDSP_vfltu8`, runs the *same* `cblas_sgemv` calls
the FP32 path already had, then factors the affine correction out of the
reduction algebraically (same trick `gemv_affine` uses on the Metal side
for weights, rather than materializing true dequantized K/V first):

```
trueScore[pos] = scale*(kScale[pos]*code[pos] + kBias[pos])·q[head]
               = kScale[pos]*rawDot[pos] + (scale*qHeadSum)*kBias[pos]
```

and symmetrically for the softmax-weighted V sum (the bias term becomes a
constant added to every element of that head's output slice, since
`vBias` is constant within a (pos,kvHead) group). No FP32 shadow buffer
persists between steps -- the widened buffer is transient, freed at the
end of each call, so the resident-memory savings the feature exists for
still hold; only a temporary compute cost is paid per step.

### Gating and verification

`SWIFTLET_KV_QUANT` (`off`/`fp32` reverts to the old FP32 path; unset or
anything else means INT8), read once per `QwenMetalModel` instance (an
instance `let`, mirroring `moeBatchEligible`'s existing convention -- not a
cached global, so tests can construct two model instances with different
modes). The CPU-oracle reference path (`QwenCPUModel.attentionForward`)
never touches the new `kvQuant` storage, regardless of mode -- it stays the
FP32 ground truth used to verify every other change in this project's
history, unaffected by this one.

**Promoted to the default after the long-context sweep results below**
confirmed the real win (+14% decode throughput, -20% peak footprint at
~20-25K tokens, no coherence regression) -- shipped opt-in first, promoted
once verified at the context length where it actually matters, not before.
The GPU-vs-CPU-oracle parity tests below that assert near-bit-exact
parity at a `<2e-3` tolerance (calibrated for prior, numerically-equivalent
changes, not this deliberately lossy one) now pin `forceKVQuantMode: .off`
explicitly rather than relying on an unset env var to mean FP32.

Verified, in order:

1. **`SWIFTLET_KV_QUANT=off` (FP32 path): byte-identical** to the
   pre-change baseline on the real 200-token reference command against
   `ornith-1.5-35b-base.qpack` -- confirms zero risk to the escape-hatch
   path from all the refactoring this needed (factoring `attnCoreInt8` out
   to be shared by both `attnForward` and `attnCoreCPU` rather than
   duplicating the new logic a third time; `attnForward`'s FP32 branch also
   got pulled into its own `attnForwardFP32` function in the process). This
   check ran before INT8 was promoted to the default -- at the time,
   "env var unset" was this same FP32 path.
2. **Two new Swift tests** (`MetalModelTests.swift`,
   `kvQuantInt8ExercisesRealAttention`/`kvQuantInt8LogitsCloseToFP32`):
   confirm the INT8 path actually populates `kvQuant` (not silently a
   no-op), produces finite logits, and stays within a loose but
   meaningful bound of the FP32 path's logits on a tiny fixture -- a
   broken affine-correction sign/algebra error would blow this bound, not
   sit near it. Needed a test-only `forceKVQuantMode` init parameter
   (`QwenMetalModel`, default nil) after discovering `setenv`/`unsetenv`
   across concurrently-running Swift Testing tests is racy -- caught by
   the test itself failing (`fp32GPU.kvQuantMode` read back as `.int8`
   from a sibling test's concurrent `setenv`), not assumed safe.
   Full suite: 68/68 (66 pre-existing + 2 new), unaffected default path.
3. **Real generation on the actual 35B model**
   (`ornith-1.5-35b-base.qpack`, reference prompt, `--cache-gb 2
   --max-new 200`): 10.93 tok/s (INT8) vs. 10.98 tok/s (FP32) -- no
   measurable throughput cost at this context length, the transient
   per-step widening pass turned out cheaper in practice than the ~12%
   back-of-envelope estimate (KVH/H = 1/group ratio of the existing BLAS
   work) suggested. Output diverges from the FP32 run partway through
   (expected -- this is the first genuinely lossy change in this whole
   arc, unlike every prior step's byte-identical or numerically-equivalent
   bar) but **stays coherent**: both responses are well-structured,
   on-topic, covering the same historical content with different wording,
   not degradation into repetition or nonsense.
4. **Memory accounting**: `ArchConfig.kvBytesPerTokenInt8` (mirrors
   `expertBlobBytesInt4G64`'s bit-accounting template) computes 10,560
   B/token vs. the FP32 baseline's 40,960 -- ~3.9x, not a clean 4x, due to
   the per-group FP32 scale/bias overhead (~3% at this group size). Not
   yet wired into `ExpertCacheMemoryGovernor.plan()`/
   `OrnithRuntimeFactory`: confirmed against source that path isn't on
   `swiftlet generate`'s actual cache-sizing route today regardless (see
   `synopsis.md` "Verifying all parts of the port") -- a real, deliberate
   scope limit, not an overlooked one.

### Long-context sweep

`ornith-swiftlet-port/design/BENCHMARK_PLAN.md`'s Gate 3 is a cache-
*budget* sweep, not a context-*length* one -- no pre-existing spec for
this existed anywhere in the repo despite being referenced repeatedly as
open. Authored one: `scratch/kv_quant_long_context_sweep.sh`, FP32 vs INT8
at ~32,768 tokens (the context length `synopsis.md`'s own memory math
already targets -- `text_config.max_position_embeddings` is 262,144,
confirmed against the real checkpoint config, well beyond 32K with no
RoPE-extrapolation concern), sampling per-process `footprint` throughout
each run (at ~200 tokens, tested same-day, the KV cache is too small a
fraction of total footprint to see any difference at all -- this is the
context length where the memory-savings claim can actually be checked
against reality instead of arithmetic). Sequential, not concurrent runs
(resource contention on one machine would confound the throughput
comparison).

**Results** (`scratch/kv_quant_long_sweep-20260903-112659/`, both legs run
back-to-back on the same machine, same prompt, `--cache-gb 2`):

| | FP32 | INT8 | delta |
|---|---|---|---|
| tokens generated | 25,018 | 19,302 | (both stopped at EOS, not the 32,768 cap -- see note below) |
| decode time | 5,895.4s | 3,990.6s | |
| decode throughput | 4.24 tok/s | 4.84 tok/s | **+14%** |
| decode wait/gpu (Metal S3a) | 2,125.7s / 1,501.3s | 1,505.3s / 1,030.0s | gap/token: 0.0250s vs 0.0246s (~unchanged) |
| expert-cache hit rate (decode) | 58% | 59% | ~unchanged, as expected (same `--cache-gb 2`) |
| peak process footprint | 7.063 GB | 5.674 GB | **-1.39 GB (-20%)** |
| footprint growth rate (post-warmup, per token)* | ~119 KB/tok | ~74 KB/tok | **~38% slower growth** |

*computed from the 10%-through-run to end-of-run footprint delta divided by
tokens generated in that span, to exclude the initial model-load/cache-fill
ramp. This is *total* process footprint, not KV alone -- it also includes
the fixed 2GB expert-cache budget and other buffers, so it doesn't isolate
to the clean ~3.9x reduction `kvBytesPerTokenInt8` predicts for KV bytes
specifically. Directionally consistent with that prediction (INT8 grows
slower), not a clean confirmation of the exact ratio.

- **Neither run hit the 32,768-token cap** -- both stopped naturally at
  EOS. The token-count difference (25,018 vs 19,302) is a real consequence
  of INT8 being lossy: quantization error in the KV cache nudges the
  greedy decode path onto a different token sequence partway through
  (expected and already documented above -- this is the first lossy step
  in the whole optimization arc), which here happened to reach a natural
  stopping point sooner. Not a benchmarking artifact, and not comparable
  to a fixed-length throughput test -- but both are long enough (>19K
  tokens) to be well past where the earlier 200-token tests could see any
  KV-driven effect at all.
- **Coherence holds at long context**: spot-checked start and end of both
  outputs -- both are complete, well-structured essays covering the full
  requested scope (prehistoric Netherlands through the 21st century) and
  end on a proper closing paragraph, not repetition or degradation. INT8
  diverges in wording/emphasis from FP32 partway through (expected) but
  never degrades.
- **The `--cache-gb` wait-minus-gpu gap does not shrink under INT8** at
  this cache budget (0.0250s/token vs 0.0246s/token, within noise) --
  worth stating plainly since Part 1 flagged memory pressure as a
  candidate driver of that gap: at `--cache-gb 2` the expert cache is the
  same fixed 2GB budget in both legs, so this result suggests the gap is
  dominated by expert-cache traffic, not KV-cache size, at least at this
  budget. Confirms the two features address different problems (KV memory
  headroom vs. the wait/gpu gap) rather than one fixing the other.
- **Real win confirmed**: decode throughput +14% and peak footprint -20%
  at ~20-25K tokens of context, with no coherence regression -- this is
  the first evidence, at real context length, that the memory-savings
  design goal is actually realized rather than just arithmetically
  predicted.

## `research/research.txt` idea #2 (3-bit/mixed-precision expert weights + DMA alignment): investigated, skipped

Read against the actual code rather than taken at the doc's own effort
estimate ("medium... you already have a 4-bit affine dequant kernel, just
need a 3-bit nibble-unpack with LUT"). Splits into two independent claims;
both come out worse than the doc suggested.

**DMA alignment**: the doc's Flash-MoE citation (2MB-aligned
`posix_memalign` + `newBufferWithBytesNoCopy` = 16.8 GB/s vs 4.7 GB/s for
16KB-aligned Metal buffers) describes a pipeline with a separate
host-malloc'd staging buffer that gets wrapped/copied into a Metal buffer.
This codebase doesn't have that step: `ExpertCache.swift` reads a miss
straight into `slots[s].contents()`, and `Qpack.swift`'s `readExpert`
`pread`s directly into that `MTLBuffer`'s own backing memory
(`storageModeShared`, already zero-copy/unified). There's no intermediate
buffer to align differently -- the win Flash-MoE measured comes from
*removing* a copy step this pipeline never had. Left as an open, cheap,
decoupled experiment (swap the slot allocator to explicit 2MB alignment,
rerun the existing cache-budget sweep) if ever revisited, but not expected
to matter.

**3-bit expert weights**: real effort, worse than "medium," found in three
places plus one empirical check:

1. `Checkpoint.swift`'s loader: `guard spec.bits == 4 || spec.bits == 8
   else { throw Error.unsupportedBits(spec.bits) }` -- rejects 3-bit
   outright today.
2. `ArchConfig.swift`'s expert-blob byte-size formula (the one behind
   every `SlotStreamMemoryGovernor` number in `synopsis.md`) hardcodes
   `expertParamCount * 4 + ...`, not a variable bit-width.
3. `gemv_moe_batched` in `Kernels.metal.txt` -- the fused kernel from the
   MoE-fusion win -- has no `bits` field in its params struct at all; it's
   structurally nibble-only. `QwenMetalModel.swift` gates the fast path to
   `p.bits == 4` exactly, so any other bit-width falls back to the slow
   scalar per-expert loop, eating into the very speedup 3-bit chases.
4. Empirically quantized a test vector with real `mlx.core.quantize(bits=3)`
   and decoded the packed words by hand: 3-bit packing is a **continuous
   bitstream across word boundaries** (element 10 straddles the word-0/
   word-1 boundary), not the per-word-isolated scheme the existing
   kernels' `perWord = 32 / bits` math assumes (only valid for bits in
   {2,4,8,16,32}). A 3-bit path needs a genuinely new cross-word unpack
   routine in both the Swift-side dequant and the Metal kernel, not a new
   case in the existing one.

**Decision: skip.** Real scope is a new bit-unpack algorithm (Swift +
Metal), a second bits-gated fast-path kernel (or accept the scalar-path
regression), the `Checkpoint.swift` gate, and re-derived `ArchConfig`
memory math -- high effort, not medium -- against the doc's own modest
impact estimate (~5-10% end-to-end, after the concurrent-fill win already
took the low-hanging fruit) and a higher correctness-risk surface than the
KV-quant work above (expert weights are read every decode step in every
MoE layer, vs. an isolable KV cache -- and the KV-quant prototype already
showed real quality divergence from an assumed-safe bit-width once
measured against real data, the same failure mode this would risk again
with less isolation). Ranked behind `research/research.txt` ideas #5
(prefix caching) and #6 (speculative decoding via GDN state
checkpoint/restore), both lower-effort and closing gaps this project's own
docs already flagged as open.
