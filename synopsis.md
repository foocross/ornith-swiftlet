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

- **CRACK abliteration damages code generation.** Verified by running the
  identical Space Invaders prompt against the official non-abliterated
  `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` checkpoint through the same
  Swiftlet build/serve/decode path. The official checkpoint produced 2676
  tokens with a complete `<script>` block containing real game logic
  (canvas setup, keyboard input, `requestAnimationFrame` game loop, HUD).
  CRACK produced 576 tokens: broken CSS, no `<script>` tag at all, and the
  model narrating its own failure. This is a property of the CRACK
  fine-tune, not a bug in the conversion pipeline or Swiftlet -- the two
  tensor bugs documented in "Two real bugs" above were verified against prose
  only and remain correctly fixed. `test-handoff.md` has the full
  experiment setup and results; it can be deleted now that the question is
  resolved.
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
- `--cache-gb` above 2 is still not a lever (see `CONVERSION_PLAN.md`
  "`--cache-gb` sweep: not a lever here" and its follow-up
  "`--cache-gb` gap revisited"): raising it grows hit rate a lot but decode
  wall barely moves, because a "wait-minus-real-exec" gap grows alongside
  the larger resident-buffer set (2.76s at 1213 slots -> 4.83s at 4854
  slots) and offsets the fill savings. A follow-up session tested the
  leading hypothesis (Metal per-resource hazard-tracking overhead scaling
  with slot count, fixable via `.hazardTrackingModeUntracked`) with a real
  A/B measurement, not just reasoning -- and the hypothesis was rejected:
  the gap grew almost identically with or without it, slightly *worse* at
  the largest cache size across a repeat run. Change reverted, no flag left
  behind. Memory pressure on this 18GB-unified-memory machine is the
  remaining, untested explanation.
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

**Follow-up session (same date):** two speculative-decoding options (MTP,
using the GGUF's own dropped `blk.40` head; DFlash, a diffusion-drafter
technique) were evaluated and neither was pursued this session -- MTP is
architecturally plausible but blocked on a decode-loop this codebase
doesn't have yet and an unresolved GatedDeltaNet-state-rollback question;
DFlash needs a trained drafter model that doesn't exist for this checkpoint.
Revisited the `--cache-gb` "not a lever" finding with a concrete mechanism
(Metal hazard-tracking overhead) and tested it directly rather than leaving
it as an open theory -- see the new "Known gaps" bullet above and
`CONVERSION_PLAN.md` "`--cache-gb` gap revisited" for the full A/B
measurement. Rejected with real evidence, change reverted; no throughput
change from this session, but the conclusion is now backed by an experiment
instead of an assumption.

## Base model build (2026-09-02): `scratch/ornith-1.5-35b-base.qpack`

Everything above through "Known gaps" is about the CRACK (abliterated)
build specifically. Separately, the plain base model --
`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`, the same official checkpoint used
throughout this document as the ground-truth reference for verifying the
CRACK tensor conversion and for the Space Invaders code-generation
comparison -- was built directly with:

```sh
swiftlet-repack --from-hf ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit \
  --output scratch/ornith-1.5-35b-base.qpack
```

No GGUF dequantization/requantization round-trip needed (`ornith-ai`
publishes this checkpoint already in the mlx-lm runtime layout
`swiftlet-repack --source` expects), so neither of the "Two real bugs"
above applies -- this path never touches the CRACK conversion script at
all. See `CONVERSION_PLAN.md` "Base model build" for the command and
rationale.

**Status: done, verified working.** `swiftlet-repack --from-hf` streamed
and repacked the checkpoint directly (18GB qpack, 547.7s wall). Sanity-run
with `swiftlet generate scratch/ornith-1.5-35b-base.qpack --gpu --chat
--cache-gb 2 --max-new 60`: loaded clean, produced coherent, on-topic
output (the model's usual reasoning-trace style, "Let me think about what
I know about this topic... Basic idea: ..." for a mixture-of-experts
explanation prompt) -- no token salad, unlike the CRACK build's first
attempt. Decode: 60 tokens in 5.4s (**11.17 tok/s**, 55-56% expert-cache
hit rate at `--cache-gb 2`), noticeably faster than the CRACK build's
original 4.9-7.8 tok/s baseline, consistent with the decode-throughput
work landed since that baseline was measured (MoE kernel fusion, CPU
attention-core vectorization, heap-based LFU eviction -- see the sections
above). `scratch/ornith-1.5-35b-base.qpack` is now the base-model
counterpart to `scratch/ornith-1.5-35b-crack.qpack`; use it for anything
that wants the non-abliterated model (e.g. code generation, per the CRACK
regression noted in "Known gaps" above).

## Memory-pressure diagnostic + KV-cache INT8 quantization (2026-09-03)

Two follow-ups from a deep-research pass (`research/research.txt`) chosen
by the user out of 9 candidate ideas, both closing questions this project's
own docs had already left open rather than starting speculative new work.

**Memory-pressure diagnostic**: the "`--cache-gb` gap" (GPU wait-minus-exec
time growing with expert-cache slot count) had a hazard-tracking-mode
explanation already tested and rejected in an earlier session, leaving
real memory pressure on this 18GB unified-memory machine as the untested
theory. Instrumented with `vm_stat`/`footprint` sampling across
`--cache-gb` 2/4/6/8 (`scratch/memory_pressure_sweep.sh`, against
`base.qpack` since `crack.qpack` no longer exists on disk). Disk swap
ruled out; memory-compressor churn (a distinct macOS mechanism) tracked
the gap's growth -- a real, if not perfectly clean, confirmation. Details:
`CONVERSION_PLAN.md` "Memory-pressure hypothesis".

**KV-cache INT8 quantization, now the default.** The FP32 K/V cache for
the 10 full-attention layers (the only per-context-length-growing memory
cost, `QwenCPUModel.DecodeState.kv`) is quantized to INT8 by default:
group size = headDim, one affine scale/bias pair per (position, kvHead),
mirroring the MLX-affine convention already used for weight quantization.
`SWIFTLET_KV_QUANT=off`/`fp32` reverts to the old FP32 path. Real-data
prototyping against actual dumped K/V (`scratch/kv_quant_prototype.py`)
found the plan's original "start with INT4" recommendation had a real
~10-20% quality cost at per-token granularity; INT8 measured near-lossless
(cosine similarity >0.9999), so INT8-only is what shipped, confirmed with
the user after the data contradicted the original plan.

Verified at short context first (68/68 unit tests, byte-identical FP32
output when reverted, coherent INT8 output), then -- since the KV cache is
too small a fraction of total footprint at ~200 tokens to see any memory
effect -- at the long-context scale where it actually matters:
`scratch/kv_quant_long_context_sweep.sh` ran FP32 vs INT8 sequentially at
`--cache-gb 2`, each generating ~20-25K tokens (both stopped naturally at
EOS, well short of the 32,768-token cap). Result: **+14% decode throughput,
-20% peak process footprint, no coherence regression** at real long-context
scale. One clarifying negative finding: the `--cache-gb` gap above did
*not* shrink under INT8 at a fixed cache budget, suggesting it's driven by
expert-cache traffic rather than KV-cache size, at least at `--cache-gb 2`.
Full numbers: `CONVERSION_PLAN.md` "Long-context sweep".

Not wired into `ExpertCacheMemoryGovernor`'s cache-sizing math -- confirmed
that governor isn't on `swiftlet generate`'s actual cache-sizing route
today regardless (see "Verifying all parts of the port" above), so this is
a deliberate scope limit. `ArchConfig.kvBytesPerTokenInt8` exists for
accounting and future wiring.

## Expert I/O hint (`F_NOCACHE`), now the default (2026-09-03)

Closes the rest of research idea #4 (the memory-pressure diagnostic above
only instrumented and explained the `--cache-gb` gap; this tests an actual
fix). `Qpack.swift`'s `QpackExpertReader` now sets `F_NOCACHE` on every
`packed_experts` fd by default -- avoids double-caching hot expert blobs
in both the OS's page cache and `ExpertCache`'s own resident
`.storageModeShared` buffers, the same redundant-caching shape behind the
confirmed compressor-churn mechanism. Measured directly against this
pipeline (not assumed from the research doc's differently-sourced claim
that no I/O hint helps): a real, reproducible **~14-15% decode-throughput
win at `--cache-gb 2`** (the shipped default; 3 clean repeats each side,
byte-identical output), a small ~2-4% regression at `--cache-gb 8` (already
documented as not a useful setting here) -- net win taken as the new
default, `SWIFTLET_EXPERT_NOCACHE=0` reverts. Full numbers and the two
`research/research.txt` idea #4 items deliberately declined (2MB-aligned
buffers -- doesn't apply, no copy step to remove; deleting the wired
expert cache to trust the OS page cache entirely -- would abandon the
bounded-memory guarantee `ExpertCache` exists for, including for iOS
jetsam avoidance): `CONVERSION_PLAN.md` "Expert I/O hint sweep".

## Research idea #2 (3-bit expert weights + DMA alignment): investigated, skipped

Checked against the actual code, not the research doc's own effort
estimate: 3-bit expert quantization is gated out today at three separate
layers (`Checkpoint.swift`'s loader rejects non-4/8-bit specs,
`ArchConfig.swift`'s memory-governor math hardcodes 4-bit byte sizing, and
the fused `gemv_moe_batched` MoE kernel has no `bits` field at all -- it's
structurally nibble-only, with 3-bit falling back to the slow scalar path
otherwise), and MLX's real 3-bit packing (confirmed by quantizing a test
vector and decoding the packed words by hand) is a continuous cross-word
bitstream, not the per-word-isolated scheme the existing dequant math
assumes -- a new unpack algorithm, not a new case. The DMA-alignment half
doesn't apply either: expert-cache fills already `pread` straight into the
`MTLBuffer`'s own backing memory with no intermediate staging buffer to
realign. High effort/risk against a modest (~5-10%) claimed payoff; ranked
behind ideas #5 and #6. Full writeup: `CONVERSION_PLAN.md` "`research/
research.txt` idea #2 ... investigated, skipped".

## Router-aware expert prefetch (2026-09-03), now the default

`research/research.txt` idea #3, chosen as the next lever after the
memory-pressure/KV-quant/`F_NOCACHE` work above. Predicts layer `i+1`'s
likely experts from layer `i`'s own already-finalized hidden state (via
`i+1`'s real router weight -- a zero-training, zero-calibration
heuristic), and prefetches them into `ExpertCache` while `i+1`'s own GPU
dispatch runs, instead of waiting for `i+1`'s real router output the way
every fetch did before. Phase 0 (pure measurement, no cache changes)
validated the predictor against the real model first -- ~79% hit@top-8,
~94% hit@top-16, far above chance -- before Phase 1 built the riskier
part: making `ExpertCache` safe for a background `prefetch()` racing the
real, synchronous `buffers()` (a new lock + per-slot in-flight-fill
tracking; a real lost-update race in the diagnostic counters, caught by a
dedicated adversarial stress test, was found and fixed along the way).
Real fetch/compute is provably untouched by prediction -- prefetch can
only waste bandwidth, never change output -- verified both by a strict
byte-identical unit test and a real-model CLI transcript diff.

A/B against the real model (`scratch/expert_prefetch_sweep.sh`, 3 clean
repeats/cell): **+4.7% at `--cache-gb 2`, +11.9% at `--cache-gb 4`,
+26.0% at `--cache-gb 6`** -- the win grows with cache size rather than
shrinking, the opposite shape from the earlier-documented "`--cache-gb`
gap," suggesting this may be what finally makes `--cache-gb` a real lever
(not yet re-measured at higher budgets to confirm). Shipped as the new
default; `SWIFTLET_EXPERT_PREFETCH=0` reverts. Full writeup, including the
counter-race bug and its fix: `CONVERSION_PLAN.md` "Router-aware expert
prefetch" and its "Phase 1: real prefetch, now the default" subsection.
That A/B ran under `F_NOCACHE` on, since reverted (see next entry) --
prefetch's relative win survives the revert (+2-3% re-verified at
`--cache-gb 2`, +10.4% at `--cache-gb 8`) but the table's exact magnitudes
predate it; not fully re-swept.

## `F_NOCACHE` default reverted (2026-09-03), same session

Investigating why this session's decode throughput measurements kept
coming in at roughly a third of this document's own documented numbers
(~3.6 vs ~11.17 tok/s, same qpack/`--cache-gb 2`) led to isolating the
cause to one setting: `F_NOCACHE` (shipped default-on earlier this
session -- see "Expert I/O hint" above), on this machine's current state,
now costs throughput rather than helping it (**11.53 tok/s off vs 3.63
tok/s on** at `--cache-gb 2`, byte-identical output either way; also
worse, not just marginally, at `--cache-gb 8`). Confirmed the regression
predates and is independent of router-aware expert prefetch by `git
stash`-ing this session's changes and retesting the exact prior commit
directly. Root cause of *why* this machine's I/O got more expensive than
when `F_NOCACHE` was originally measured was not tracked down (a stray
17-hour orphaned server process was found and killed but didn't
meaningfully help; an long-running MLX server process with a real ~3.6GB
resident footprint was noted but not tested in isolation -- pausing other
processes to test that was correctly blocked by the permission system).
**Reverted to default off**; `SWIFTLET_EXPERT_NOCACHE=1` opts back in.
75/75 tests pass. Worth a proper re-sweep on a quiet machine -- full
detail: `CONVERSION_PLAN.md` "`F_NOCACHE` default reverted".

## oMLX server ruled out as the throughput-drop cause (2026-09-04)

The user reported ~14 tok/s previously vs. ~11 tok/s now. The ~14 figure
traces to the original `F_NOCACHE` sweep above (13.3-13.5 tok/s at
`--cache-gb 2`); current numbers match the *reverted* default's
~11.17-11.76 tok/s range documented above -- so the drop is the already-
known, already-explained `F_NOCACHE` regression-and-revert, not a new
bug. The one loose thread that section left open -- a long-running `oMLX`
server process (~3.6GB resident, 12 days uptime) noted but never tested in
isolation -- was tested directly this session (user killed it): **no
measurable effect** (10.70 tok/s before, 10.76-10.91 after, 3 repeats).
Both current defaults (prefetch on, `F_NOCACHE` off) were reconfirmed
correct on this machine at the same time. Root cause of the machine-wide
slowdown between the two measurement windows remains open. Full detail:
`CONVERSION_PLAN.md` "oMLX server ruled out as the I/O-cost cause".
