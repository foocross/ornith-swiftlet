# Running Ornith-1.5-35B-A3B-CRACK (GGUF) on Swiftlet: synopsis

## The ask

Run `dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF`'s
`Ornith-1.5-35B-A3B-CRACK-Q4_K_M.gguf` on
[Swiftlet](https://github.com/leonickson1/Swiftlet), using the preliminary
work in `ornith-swiftlet-port` as a starting point.

## The core problem: two incompatible ecosystems

`ornith-swiftlet-port`'s overlay is a real, tested integration (27 standalone
tests + 12 overlay tests, all passing), but it was built for an **MLX**
checkpoint (`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`) repacked with Swiftlet's
own `swiftlet-repack` tool. The requested file is a **GGUF** -- llama.cpp's
own container and K-quant scheme, produced by a completely different
toolchain and, in this case, a different producer (`dealignai`'s "CRACK"
abliteration of the base `ornith-ai` checkpoint). Swiftlet has no GGUF
loader, and `dealignai` publishes no MLX/safetensors build of the
uncensored variant -- only GGUF, MXFP8, and "JANG" formats. There was no
way to point the existing overlay at the requested file directly.

Three options existed to close that gap:

1. **Convert GGUF -> MLX -> qpack** (chosen): dequantize the GGUF, rebuild an
   HF-format checkpoint, requantize with `mlx_lm`, then run the *existing*
   overlay completely unchanged.
2. Write a new pure-Swift GGUF->qpack converter (no Python dependency, more
   engineering, still a double-quantization hop).
3. Teach Swiftlet's Metal kernels to read K-quant blocks natively (biggest
   scope, a second backend, abandons the qpack scheme the whole port kit is
   built around).

Option 1 won because the local machine (Apple M3 Pro, 18GB RAM, 460GB disk)
already had every tool needed: `mlx-lm` 0.31.3 with **native
`qwen3_5_moe.py` support** (this exact hybrid GatedDeltaNet+MoE
architecture is a first-class citizen, not something that needed teaching),
Swift 6.3.3, and Python. It required zero Swiftlet source changes beyond the
overlay patches that already existed.

## Feasibility research, verified against primary sources (not assumed)

- `Sources/SwiftletRepack/main.swift` + `Sources/SwiftletCore/Qpack.swift`
  (read directly from a fresh clone): `swiftlet-repack --source <local-dir>`
  accepts any correctly-shaped **mlx-lm post-quantization runtime format**
  directory (`model.layers.{n}.mlp.switch_mlp.{gate,up,down}_proj
  .{weight,scales,biases}`, `shape[0] == numExperts`). It explicitly
  rejects raw HF checkpoints ("convert it with mlx-lm first").
- `Sources/SwiftletCore/QwenConfig.swift`: parses the exact nested
  `text_config` config.json shape `ornith-ai/Ornith-1.5-35B-A3B` already
  uses, including auto-detecting the `language_model.` weight prefix
  (`Checkpoint.swift:74-77` resolves it transparently either way).
- GGUF's arch tag is `qwen35moe` (read directly from the file's
  `general.architecture` metadata field). `gguf-py`'s `tensor_mapping.py`
  and `constants.py` give the canonical GGUF<->HF name mapping for that
  arch, and mlx-lm's `qwen3_5.py` / `qwen3_next.py` / `qwen3_5_moe.py` give
  the exact mlx-side parameter names and `sanitize()` behavior.
- Cross-checked every derived tensor name against the **real**
  `model.safetensors.index.json` of the base `ornith-ai/Ornith-1.5-35B-A3B`
  checkpoint -- all matched exactly, including the `model.language_model.*`
  -> `language_model.model.*` rewrite and the fused
  `mlp.experts.gate_up_proj` naming in the raw checkpoint (which GGUF
  turned out to store pre-split into separate `ffn_gate_exps`/
  `ffn_up_exps`/`ffn_down_exps`, so the converter targets mlx's final
  `switch_mlp.{gate,up,down}_proj` names directly rather than round-tripping
  through a fuse-then-split step it doesn't need).

## The pipeline (implemented, working)

```
Ornith-1.5-35B-A3B-CRACK-Q4_K_M.gguf   (llama.cpp K-quant, arch "qwen35moe")
        | scratch/build_hf_checkpoint.py: dequantize (gguf python pkg)
        | + rename tensors to raw HF convention + fix two GGUF/mlx-lm
        | convention mismatches (see "Two real bugs" below)
        v
local HF-format checkpoint (bf16 safetensors, ~65GB, config.json copied
verbatim from ornith-ai/Ornith-1.5-35B-A3B -- same architecture, CRACK only
changes weight values)
        | python -m mlx_lm convert -q --q-bits 4 --q-group-size 64
        |   --q-mode affine --dtype bfloat16
        v
mlx-lm runtime-format checkpoint (~18GB, switch_mlp.*.{weight,scales,biases})
        | swiftlet-repack --source <dir> --output out.qpack
        v
.qpack container (~18GB)
        | (Swiftlet already cloned with ornith-swiftlet-port's overlay
        |  patches 0001+0002 applied -- additive only, one line changed
        |  in ArchConfig.swift)
        v
.build/release/swiftlet generate out.qpack --gpu --chat --cache-gb 2 \
  --prompt "..."
```

Text-only, K=8 baseline: the converter drops GGUF's `blk.40` block (llama.cpp
stores the model's one-layer MTP speculative head there -- confirmed by its
`nextn.{eh_proj,enorm,hnorm,shared_head_norm}` tensors) and there's no vision
data in this GGUF to worry about (it lives in the separate `mmproj-*.gguf`).
This matches the scope `ornith-swiftlet-port`'s overlay already assumed
before any of this work started, not a new limitation introduced here.

## Two real bugs, found by verifying against ground truth

The first full run through the pipeline **loaded, ran at a normal token
rate, and reported a plausible 56-62% expert-cache hit rate -- and produced
complete token salad.** That's the headline lesson: a 35B-parameter MoE
model with a subtly wrong linear-attention state doesn't crash or look
obviously broken. It runs. It just generates garbage. "The pipeline
executes without error" is not evidence of correctness for something this
size; only comparing actual dequantized values against a known-good
checkpoint caught these.

The verification method: download `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`
(the real official MLX checkpoint), repack and run it through the *exact
same* Swiftlet build+overlay+Metal path used for the converted CRACK model.
It produced fully coherent output immediately -- proving Swiftlet, the
overlay, and the qpack/Metal pipeline were all correct, and the bug had to
be in the conversion script specifically. Then: dequantize matching tensors
from both checkpoints and diff them elementwise.

1. **`ssm_a` (GGUF) is `-exp(A_log)`, not `A_log`.** GGML's SSM convention
   stores the ready-to-use decay term; mlx-lm's `GatedDeltaNet` wants the
   log-parameterized form and re-exponentiates it internally. Fix:
   `A_log = log(-ssm_a)`.

2. **A GQA-style head-grouping mismatch between llama.cpp and mlx-lm.** This
   architecture has 16 key-heads and 32 value-heads (a 2x expansion, same
   shape as ordinary grouped-query attention). llama.cpp's GGUF export lays
   the 32 value-heads out **split**: `[head0..head15 "primary"][head0..head15
   "secondary"]`. mlx-lm's `GatedDeltaNet` expects them **consecutive per
   key-head**: `[kh0_v0, kh0_v1, kh1_v0, kh1_v1, ..., kh15_v0, kh15_v1]`.
   Verified exactly against the official checkpoint: `sorted(A_log)` matched
   to float32 precision (max diff 5.6e-8) -- only the per-element *order*
   was wrong. The exact permutation is
   `np.arange(32).reshape(16, 2).T.flatten()`, equivalently
   `gguf_array.reshape(2, 16).transpose(1, 0)`.

   This permutation had to be applied to **every** tensor organized by the
   32 value-heads, not just `A_log`:
   - `A_log`, `dt_bias` -- 1 scalar per head (32,) -> `reshape(2,16).T`
   - `in_proj_a`, `in_proj_b` -- 1 row per head (32, hidden) -> reorder rows
   - `in_proj_z`, the **v-slice** of `in_proj_qkv`, `out_proj` -- groups of
     128 rows/cols per head (4096, hidden) -> reorder row/col *groups*
   - the **v-slice** of `conv1d` (also groups of 128 rows) -- **missed on
     the first fix pass**. Its q/k slices matched the official checkpoint
     exactly (diff 0.0, since they're unquantized), and that's specifically
     what caught it: checking each of the three slices of a concatenated
     [q|k|v] tensor separately, rather than trusting a whole-tensor
     statistical comparison, surfaced the one slice that was still wrong
     after the first round of fixes looked complete everywhere else checked.

   `ssm_norm` (128,, the RMSNormGated weight) has no head axis and needed no
   fix -- confirmed identical to the official checkpoint from the start.

Every other tensor category was cross-checked the same way and found
correct on the first pass: token embeddings (248,070 ids compared 1:1
against the GGUF's own embedded tokenizer -- zero mismatches), norm
weights, full-attention q/k/v/o/q_norm/k_norm (including the q_proj
query/gate fused-output split, which does *not* need reordering since
num_attention_heads matches 1:1, no expansion), the MoE router,
shared-expert, and routed-expert (switch_mlp) weights.

A separate, earlier false hypothesis worth recording: mlx-lm's
`TextModel.sanitize()` has a `+1.0` correction for certain norm weights,
gated on whether `conv1d.weight`'s raw layout looks untransposed. It was
tempting to assume this always fires for a "genuine" checkpoint. It doesn't
-- direct inspection of real dequantized norm values (mean ~0.9-2.6, not
~0) showed they're already in final form, so the correct move was to
pre-shape `conv1d` into mlx's native layout ourselves and bypass that
auto-correction, not rely on it firing.

## Verifying "all parts of the port," not just applying the patch

Applying the overlay's patch files and having them compile is not the same
as exercising what they do. Checked separately:

- `swift test` in the standalone `ornith-swiftlet-port` package: 27/27 pass.
- `swift test --filter OrnithSupportTests` in the patched Swiftlet clone:
  12/12 pass (architecture validation, qpack shape-drift/manifest
  rejection, payload-size checks, memory-plan math).
- `OrnithRuntimeFactory.preflight()`/`.make()` **against the real qpack**
  (not synthetic fixtures) via a throwaway executable target, since
  `swiftlet generate`'s CLI path uses Swiftlet's generic model loader and
  never touches `OrnithRuntimeFactory`/`SlotStreamMemoryGovernor` at all --
  it happened to work regardless because Swiftlet already natively
  understands the Qwen3.5 architecture family. `preflight()` validated the
  container and turned a 4GiB total-memory target into a concrete 1.76GiB
  expert-cache budget; `make()` built a real `QwenMetalModel` through that
  validated path. This is the kit's actual differentiator (explicit
  architecture guarding + memory-governed cache sizing instead of a
  hand-picked `--cache-gb`), and it needed to be exercised deliberately,
  not assumed to work because the patch applied cleanly.

## Results

Coherent, correctly-formatted, on-topic generation confirmed at 80 and 200
tokens, including the model's documented `<think>...</think>` reasoning
trace before the final answer.

**Throughput** (M3 Pro, `--cache-gb 2`):

| Run | Tokens | Wall time | tok/s | Expert-cache hit rate |
|---|---|---|---|---|
| 80-token test | 80 | 16.3s | 4.9 | 56% |
| 200-token test | 200 | 25.7s | 7.8 | 60% |

The spread is cache warm-up (misses cost a disk read); longer runs trend
toward the higher end as more of the working set lands in cache.

**Memory for a 32K-token context**, from the real `SlotStreamMemoryGovernor`
math (not hand-derived):

| Component | Size | Scales with |
|---|---|---|
| Dense/resident weights | 1.29 GiB | fixed |
| DeltaNet fixed state (FP32 recurrence + conv history) | 62.8 MiB | fixed |
| KV cache @ 32K tokens (40,960 B/token, full-attention layers only) | 1.25 GiB | context length |
| Scratch reserve | 256 MiB | fixed |
| Safety margin | 8% of target | target memory |

Minimum viable total-memory target at 32K context: **~3.13 GiB** (the
governor refuses below that rather than silently starving the cache).
Above the floor, everything extra is expert cache:

| Target memory | Expert cache | Resident expert slots (of 10,240 total) |
|---|---|---|
| 4 GiB | 0.82 GiB | 500 |
| 6 GiB | 2.66 GiB | 1,616 |
| 8 GiB | 4.50 GiB | 2,733 |

## What's kept, what's reusable

- `scratch/ornith-1.5-35b-crack.qpack` (18GB) -- the runnable model.
- `scratch/build_hf_checkpoint.py` -- the GGUF->HF converter, with both
  fixes and their derivations documented inline. Reusable for any future
  GGUF drop of this same architecture family (`qwen35moe`), including
  presumably the plain (non-abliterated) Ornith 1.5 35B GGUF or other
  quant levels of the CRACK build, since the bugs fixed are properties of
  the *architecture*, not this specific quantization or fine-tune.
- `CONVERSION_PLAN.md` (project root) -- shorter status/runbook version of
  this document, written mid-process.
- Everything else (the 21.7GB GGUF, the ~65GB intermediate HF checkpoint,
  diagnostic downloads of the official checkpoint) was deleted once no
  longer needed -- disk was genuinely tight (460GB total, needed careful
  sequencing) and none of it is needed to reproduce the result, only the
  script and the final qpack are.

## Known gaps (inherited from the original overlay scope, not new)

- Text-only, native K=8. No MTP speculative decoding (GGUF's `blk.40` is
  dropped), no vision (not present in this GGUF regardless).
- ~~Swiftlet's expert-cache fill path is still serial reads per miss; the
  overlay deliberately didn't change that without trace evidence, per its
  own design notes.~~ Traced and fixed (see `CONVERSION_PLAN.md`, "Decode
  throughput"): serial fill measured at ~46% of decode wall time, fully
  idling the GPU; bounded concurrent pread per miss batch took decode from
  5.79 to 8.89 tok/s at `--cache-gb 2` with byte-identical output. A
  follow-up pass fixed a measurement bug (prefill's expert-cache stats had
  been folded into decode's) and then a real O(slots) linear scan in LFU
  eviction (now a min-heap), for a running total of ~9.98 tok/s (~72%
  cumulative). Not yet re-swept across other cache budgets or longer
  contexts.
- The reported tok/s figures are from two short runs (80 and 200 tokens),
  not a proper sweep across cache budgets and prompt lengths -- directional,
  not a benchmark.

## Status as of 2026-09-02: decode-throughput optimization, MoE fusion + CPU vectorization landed

**See `handoff.md` for the full handoff to continue this** -- exact repo/
commit state (two local git repos, `Swiftlet/` and this root, nothing
pushed), what was tried and rejected and why. Short version: decode
throughput went from 5.79 to ~9.4-9.94 tok/s (`--cache-gb 2`, same reference
qpack/prompt) across three real fixes (concurrent expert-cache fill,
corrected a measurement bug, heap-based LFU eviction), then a further +5-6%
from MoE kernel fusion (a new bindless batched-GEMV Metal kernel collapsing
`encodePendingMoE`'s up to 38 per-MoE-layer dispatches down to 9), then a
further +4.3% (200 tokens) to +11.4% (354 tokens) from vectorizing the CPU
attention core (`cblas_sgemv`/`vDSP` replacing scalar Swift loops) -- all
verified byte-identical output at every step except the last, which is
numerically equivalent (not bit-identical, due to Accelerate's different
float-reduction order) but produced byte-identical *generated text* in
practice on both paired runs. A cache-budget sweep and a fill/GPU-dispatch
overlap idea were both investigated and rejected with real evidence (see
`CONVERSION_PLAN.md` "Decode throughput" for the numbers on both); MoE
fusion was investigated, pursued, and landed (see `CONVERSION_PLAN.md` "MoE
kernel fusion" and `handoff.md`) -- a real win, smaller than the ~2x
total-dispatch-count reduction suggested, since most of decode's GPU time
turned out to be real compute rather than per-dispatch overhead. The CPU
attention-core vectorization (see `CONVERSION_PLAN.md` "CPU attention-core
vectorization") is the one fix in this whole arc that targets a cost which
scales with context length rather than staying fixed -- it's expected to
matter more, not less, at the longer contexts this project hasn't yet
benchmarked. The cache-budget and longer-context sweeps
`design/BENCHMARK_PLAN.md` calls for remain open; the longer-context sweep
in particular is now the natural next step to get real numbers on that
growth curve rather than the two-point comparison done so far.
