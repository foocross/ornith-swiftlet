# Handoff: Ornith-1.5-35B decode-throughput optimization

Written for the next session/model picking this up. Read `synopsis.md` first
(what this project is, how the GGUF got onto Swiftlet at all), then
`CONVERSION_PLAN.md` "Decode throughput" section (the full numbers this
document summarizes), then this file for exactly where things stand and what
to do next.

## Status as of 2026-09-03 (later session) -- read this section first, rest of file is dated history

Everything below "Where things stand right now" was written by an earlier
session and is stale in its specifics (still cites `crack.qpack`, which no
longer exists on disk; doesn't know about any of what follows) but still
accurate as *history* -- keep it for the narrative, don't trust its numbers
as current.

**Shipped since then** (full detail: `synopsis.md` and `CONVERSION_PLAN.md`,
both updated same-session):
- **Router-aware expert prefetch**, now the default (`SWIFTLET_EXPERT_PREFETCH=0`
  reverts). Predicts layer `i+1`'s experts from layer `i`'s own hidden state
  via `i+1`'s real router weight, prefetches into `ExpertCache` while `i+1`'s
  GPU dispatch runs. New `ExpertCache` locking + in-flight-fill tracking so a
  background `prefetch()` can safely race the real `buffers()`. A real
  lost-update race in the diagnostic counters was found by a dedicated
  adversarial stress test and fixed.
- **`F_NOCACHE` default reverted** (was on, now off; `SWIFTLET_EXPERT_NOCACHE=1`
  opts back in). A prior session's shipped default turned out to cost ~3x
  decode throughput on this machine's current state (11.5 vs 3.6 tok/s at
  `--cache-gb 2`, isolated via `git stash` + rebuild-and-retest of the exact
  prior commit) -- root cause of *why* not identified, just reverted on clean
  A/B evidence.
- Best current measured decode throughput: **11.76 tok/s** (`--cache-gb 2`,
  `F_NOCACHE` off, prefetch on -- both now defaults), on a machine whose
  absolute numbers were never fully quiet/characterized this session. Treat
  as directional, re-sweep on a quiet machine before citing precisely.
- **Repo state**: both repos have uncommitted working-tree changes from this
  work (`Swiftlet/`: `ExpertCache.swift`, `QwenMetalModel.swift`, `Qpack.swift`,
  `SwiftletCLI/main.swift`, `MetalModelTests.swift` modified, new
  `ExpertCacheTests.swift`; root: `CONVERSION_PLAN.md`/`synopsis.md` modified,
  `scratch/expert_prefetch_sweep.sh`/`scratch/router_overlap_analysis.py` new).
  Nothing committed yet -- same "don't push, ask before committing" convention
  as below.

**Next task, scoped below: multi-token prediction (MTP) speculative
decoding.** See "MTP investigation" section near the end of this file --
skip straight there if that's what you're picking up.

## Where things stand right now

**Decode throughput: 5.79 -> ~9.4-9.94 tok/s from the fill/eviction arc,
+5-6% from MoE kernel fusion, plus a further +4-11% (grows with context
length) from vectorizing the CPU attention core, plus a further +14% at
long context (~20-25K tokens) from KV-cache INT8 quantization -- now the
default (`--cache-gb 2`, same reference qpack/prompt throughout).** Run-to-run variance on this machine is
real (thermal/load noise) -- treat anything in that range as "current
state," not a single precise number. All changes verified byte-identical
generated output against the pre-change baseline at every step (one
deliberate exception: the CPU-vectorization change is numerically
equivalent, not bit-identical, in intermediate floats -- see below); all 63
Swiftlet tests (62 + 1 new) + 27 standalone `ornith-swiftlet-port` tests
still pass.

The "pursue MoE kernel fusion" decision this file left open is now resolved
(the user chose to pursue it): `encodePendingMoE`'s per-expert GEMV/silu_mul
loops (up to 38 dispatches/MoE-layer) are now a bindless batched dispatch
(9/MoE-layer) behind `SWIFTLET_NO_MOE_BATCH=1` (default: batching on). See
`CONVERSION_PLAN.md` "MoE kernel fusion" for the full design/verification
detail -- short version: real, verified (byte-identical output, 63/63 tests
including a new numeric kernel test, `FastPathBaseline`'s dispatch-count
constants updated to match), but a smaller win than the halved dispatch
count suggested (+5-6% decode tok/s, not the higher end of the "double-digit"
estimate this file floated) -- most of decode's GPU time turned out to be
real compute, not per-dispatch overhead.

**New this session: CPU attention-core vectorization**, found by reviewing
this file + `synopsis.md` for the next lever rather than a new GPU trace.
`attnCoreCPU`/`attnForward` (`QwenMetalModel.swift`) and `rmsNorm`/
`softmaxRow` (`QwenCPUModel.swift`) were scalar Swift loops despite
`import Accelerate` sitting unused in the latter file; replaced with
`cblas_sgemv`/`vDSP` calls. See `CONVERSION_PLAN.md` "CPU attention-core
vectorization" for full detail -- short version: this is the one decode-time
cost in the whole arc that's `O(context length)` rather than `O(1)` (10 of
40 layers are full-attention, cost is `O(H*kvLen*hd)` per such layer per
step), so unlike everything else measured so far it's expected to matter
increasingly more at longer contexts, not stay fixed. Measured +4.3% at 200
tokens, +11.4% at 354 tokens (one paired comparison each, not a full sweep)
-- consistent with that growth prediction.

## Repo/commit state -- nothing pushed anywhere

Two separate local git repos, both untouched on origin:

- **`Swiftlet/`** (clone of `github.com/leonickson1/Swiftlet`, origin
  `main` at `aaa910a`): branch **`ornith-decode-throughput`**, six commits
  on top:
  1. `6d14f41` -- applies the pre-existing `ornith-swiftlet-port` overlay
     (was sitting as uncommitted working-tree changes before this session;
     unrelated to the throughput work, committed first to get a clean base)
  2. `62c30cc` -- bounded concurrent expert-cache fill (the big win, +56%)
  3. `491d437` -- fixed a measurement bug (prefill's expert-cache stats had
     been folded into decode's) + heap-based LFU eviction (+7% more)
  4. `14df4ea` -- opt-in per-category GPU timing diagnostic
     (`SWIFTLET_CATEGORY_TIMING=1`), off by default
  5. `336bdbf` -- MoE kernel fusion: batched bindless GEMV, +5-6% further
  6. `6f20f28` -- CPU attention-core vectorization (`cblas_sgemv`/`vDSP`),
     +4-11% depending on context length reached so far (grows with it)
- **`orninth/`** (this root project, freshly `git init`'d this session):
  branch `main`, two commits (`c931068` initial, `8fe969d` docs + logs).
  `.gitignore` excludes `Swiftlet/` (tracked separately above), `.build/`,
  and the multi-GB scratch intermediates (qpack, GGUF, mlx checkpoint) --
  regenerable per `CONVERSION_PLAN.md`, not needed to reproduce anything.

Neither repo has been pushed. If you make further changes, keep following
the pattern already established: verify (tests + byte-identical output
diff) before committing, one logical change per commit, don't push unless
asked.

## What was done, in order (see `CONVERSION_PLAN.md` "Decode throughput" for full numbers/reasoning on each)

1. **Gate 4 (measure, don't assume)**: instrumented `ExpertCache` to isolate
   time spent in `pread` on a cache miss from GPU wait time. Confirmed
   serial fill was ~46% of decode wall and fully idled the GPU.
2. **Gate 5 (fix it)**: bounded concurrent `pread` within one layer's
   top-K miss batch (width via `SWIFTLET_EXPERT_READ_CONCURRENCY`, default
   8). Required making `QpackExpertReader`'s fd-open thread-safe.
   **5.79 -> 8.89 tok/s.**
3. **`--cache-gb` sweep (2/4/6/8GB): tried, not a lever.** Hit rate climbed
   a lot, decode wall barely moved -- GPU-side dispatch overhead grew
   alongside the larger resident-buffer set and offset the fill savings.
   Left the default at 2.
4. **Fill/GPU-dispatch overlap in the fast path: investigated, rejected.**
   First estimated ~45% from misreading which function is the actual hot
   path. On closer reading, the real decode path
   (`QwenMetalModel.stepOneFast`/`encodePendingMoE`) already pipelines
   layer N's MoE into layer N+1's command buffer. The remaining sequencing
   is a genuine data dependency (layer N's picks come from layer N's own
   command buffer; layer N+1 needs layer N's combined output), not a
   scheduling gap. Real available overlap: ~1-2% of decode wall. Not worth
   the correctness risk of restructuring a barrier-synchronized,
   register-offset Metal kernel pipeline for that. **Left alone.**
5. **Measurement bug found and fixed**: the expert-cache stats line was
   cumulative since process start, silently mixing prefill's contribution
   into what was reported as decode's. Fixed by snapshotting counters after
   prefill.
6. **LFU eviction bookkeeping: found and fixed, real 6% win.** With the
   measurement corrected, ~13% of decode wall was still unaccounted for.
   Ruled out CLI loop overhead (argmax scan, per-step tokenizer re-decode:
   measured 0.05s, negligible). Found `ExpertCache.slotForFill`'s
   LFU-eviction victim search was an O(slots) linear scan on every miss
   once the cache filled (1,213 slots at `--cache-gb 2`, 25k+ decode
   misses) -- 1.3s, 6% of decode wall. Fixed with a min-heap over
   `(freq, lastUse)`, lazy staleness checking on pop, periodic compaction.
   **~9.27 -> 9.98 tok/s.** Pure CPU/data-structure change, no Metal
   involved; correctness never depended on which slot gets evicted, only
   speed does.
7. **GPU-exec breakdown by category: measured, real Metal timestamps.**
   With fill and bookkeeping both fixed, GPU exec (~44-49% of decode wall)
   is now the largest untouched bucket. Built an analytical byte-budget
   model first (exact tensor shapes from the qpack, bandwidth-bound GEMV
   assumption), which suggested delta-net was the biggest category. Then
   checked it against real `cb.gpuStartTime`/`gpuEndTime` per category
   (opt-in `SWIFTLET_CATEGORY_TIMING=1`, splits `stepOneFast`'s command
   buffers along category lines; off by default, verified zero effect on
   the default path). **The real measurement disagreed with the model in
   an important way:**

   | Category | Measured | Byte-budget predicted |
   |---|---|---|
   | MoE (routed+shared+router) | **46.3%** | 39.4% (+18% over budget) |
   | Delta/GatedDeltaNet | 34.5% | 34.2% (matches) |
   | lm_head (vocab=248,320) | 10.4% | 17.1% (-39% under budget) |
   | Attention | 8.8% | 9.2% (matches) |

   MoE is the category running most inefficiently relative to bytes moved
   -- consistent with it being by far the most dispatch-heavy path (8,840
   of the diagnostic run's command buffers vs. 4,420 for attention).
   lm_head's one big dispatch amortizes overhead well and needs no
   attention. **This points at MoE kernel fusion (batching the K=8
   experts' gate/up/down GEMVs into fewer, larger dispatches) as the
   best-evidenced next lever.**

## MoE kernel fusion: done, this session

The open decision this file previously left ("pursue MoE kernel fusion
next?") is resolved -- pursued, landed, verified. Summary (full detail in
`CONVERSION_PLAN.md` "MoE kernel fusion"):

- New Metal kernel `gemv_moe_batched` (`Kernels.metal.txt`) + Swift encode
  helper `MetalEngine.encodeGemvMoEBatched`/blocking test wrapper
  `gemvMoEBatchedBlocking`. First bindless (GPU-address-array) kernel in
  this codebase -- every expert's qpack blob has the same internal
  gate/up/down byte layout, so a tiny per-layer buffer of K raw
  `MTLBuffer.gpuAddress` values plus `enc.useResource` lets one dispatch
  cover all K resident experts.
- `encodePendingMoE` (`QwenMetalModel.swift`): gate+up now one dispatch
  across all K experts (was 2K), down now one dispatch (was K); `silu_mul`
  batched unconditionally (K per-expert calls -> 1, pure dispatch-count
  win, no new kernel needed -- the outputs were already contiguous by
  expert index). 38 dispatches/MoE-layer -> 9.
- Gated behind `SWIFTLET_NO_MOE_BATCH=1` (default: on), mirroring
  `SWIFTLET_NO_FAST_GEMV`'s shape; only engages in qpack+`ExpertCache`
  mode with this model's actual expert quant profile (4-bit,
  groupSize%8==0) -- the non-cache "stacks" fallback is untouched.
- Verified: new `MetalKernelTests.gemvMoEBatchedMatchesCPU` (K=3 synthetic
  blobs, dual- and single-stage, `<1e-3` vs CPU reference); full `swift
  test` (63/63) after updating `FastPathBaseline`'s hardcoded dispatch
  counts (fails by design post-change, fixed to the new real counts, not
  worked around); two paired full-35B-model runs (flag on/off), generated
  text byte-identical to `gate10_heap.out` in all four runs.
- Result: **+5-6% decode tok/s** (9.35->9.85, 9.41->9.94 across two paired
  runs), smaller than the ~2x total-dispatch-count drop
  (354,400->178,400 over the 200-step run) would suggest -- most of
  decode's GPU time is real compute, not per-dispatch overhead, more than
  the earlier byte-budget-vs-measured gap implied. A real, verified win,
  landing at the low end of the "double-digit-percent" estimate this file
  previously floated rather than the high end.

No further MoE-fusion work is planned as part of this arc; what's still
open (cache-budget sweep, longer-context sweep -- see "Key files" below)
is unrelated to this change and was already open before it.

## How to reproduce / continue benchmarking

Reference command (same qpack/prompt used for every number in this
document and in `CONVERSION_PLAN.md`):

```sh
cd Swiftlet && swift build -c release
cd ../scratch
../Swiftlet/.build/release/swiftlet generate ornith-1.5-35b-crack.qpack \
  --gpu --chat --cache-gb 2 --max-new 200 \
  --prompt "Write a short paragraph about the history of the Netherlands."
```

Stderr prints a full stats breakdown: prefill S3a/expert-cache lines, decode
tok/s, decode Metal S3a (GPU wait/exec), decode expert-cache (hits/misses/
fill/bookkeeping, decode-only -- prefill's share is snapshotted out
separately), decode loop overhead (argmax/text-decode, both negligible).

Diagnostic env vars (both opt-in, zero effect on default behavior when
unset -- verified each time they were added):

- `SWIFTLET_EXPERT_READ_CONCURRENCY=<n>` -- bound on concurrent `pread`s
  per miss batch (default 8; saturates around 4-8 for this model's K=8).
- `SWIFTLET_CATEGORY_TIMING=1` -- prints the real per-category GPU-time
  breakdown (moe/delta/attention/lm_head) at the end of a `generate` run.
  Diagnostic mode's own extra command-buffer commits make its *absolute*
  tok/s unrepresentative of production -- only the relative category
  split is meaningful.

Before trusting any further optimization: rebuild, run `swift test` (63
tests), run the reference command, and diff the generated text byte-for-byte
against a prior run's `.out` file (several are kept in `scratch/`, e.g.
`gate10_heap.out`) -- this is how every change in this arc was verified,
and it's cheap enough that skipping it isn't worth it.

- `SWIFTLET_NO_MOE_BATCH=1` -- forces `encodePendingMoE`'s original
  per-expert GEMV/silu_mul loop instead of the batched dispatch (default:
  batching on). A/B-debugging escape hatch, same shape as
  `SWIFTLET_NO_FAST_GEMV`.

The CPU attention-core vectorization has no on/off flag (unlike the levers
above) -- it's a straight numeric-equivalence swap of the reduction
implementation, not a behavioral change worth A/B-gating. Its own
verification runs are `scratch/vdsp_attn_run1.out` (200 tokens, vs.
`moefusion_batched.out`) and `scratch/longctx_old.out`/`longctx_new.out`
(354 tokens, pre-/post-change binaries via `git stash`) -- same
byte-for-byte generated-text diff method as everything else in this list,
just without a flag to toggle in one binary.

## Key files

- `Swiftlet/Sources/SwiftletCore/ExpertCache.swift` -- concurrent fill +
  heap-based LFU eviction (fill/eviction arc's core fix).
- `Swiftlet/Sources/SwiftletCore/Qpack.swift` -- `QpackExpertReader`,
  thread-safe fd-open.
- `Swiftlet/Sources/SwiftletCore/QwenMetalModel.swift` -- `stepOneFast`/
  `encodePendingMoE` is the real decode hot path (not `moeForward`, which
  looked like it at first glance but isn't what `--gpu` decode actually
  runs). Category-timing diagnostic, `drainPendingMoE` helper, and the
  MoE-fusion wiring (`moeBatchEligible`, `moeExpertBases`, the batched
  gate+up/down dispatches in `encodePendingMoE`) all live here.
  `attnCoreCPU`/`attnForward` are the vectorized (`cblas_sgemv`) attention
  cores -- the fast-path and fallback versions of the same computation.
- `Swiftlet/Sources/SwiftletCore/QwenCPUModel.swift` -- `rmsNorm`/
  `softmaxRow`, now `vDSP`-backed; this file is otherwise the CPU
  correctness oracle, deliberately left simple elsewhere ("favors clarity
  over speed" per its own doc comment).
- `Swiftlet/Sources/SwiftletCore/MetalEngine.swift` -- `gemv_moe_batched`'s
  Swift-side encode helper (`encodeGemvMoEBatched`) and blocking test
  wrapper (`gemvMoEBatchedBlocking`).
- `Swiftlet/Sources/SwiftletCore/Kernels.metal.txt` -- `gemv_moe_batched`,
  the bindless batched-GEMV kernel itself.
- `Swiftlet/Tests/SwiftletCoreTests/MetalKernelTests.swift` --
  `gemvMoEBatchedMatchesCPU`, the new kernel's numeric correctness test.
- `Swiftlet/Tests/SwiftletCoreTests/MetalModelTests.swift` --
  `FastPathBaseline`'s dispatch-count constants, updated for the new
  batched dispatch counts (split into `q4Baseline`/`q35Baseline`, non-cache
  fixtures, vs. `q4StreamingBaseline`, the real `ExpertCache` path, since
  they now diverge).
- `Swiftlet/Sources/SwiftletCLI/main.swift` -- stats-line instrumentation
  from the fill/eviction arc.
- `CONVERSION_PLAN.md` -- the full numbers/narrative for every step above,
  written as it happened; the primary source, this file is a summary of it.
- `ornith-swiftlet-port/design/BENCHMARK_PLAN.md` -- the pre-existing Gate
  0-5 plan this whole arc followed; Gate 3 (cache-budget sweep across a
  full corpus) and a longer-context sweep are still open per that plan,
  and unaffected by the MoE-fusion work above.

## Also investigated this session: CRACK code-generation test, `--cache-gb` gap

**CRACK code-generation bug: confirmed as abliteration damage, not a
conversion bug.** Ran the Space Invaders prompt against the official
non-abliterated `Ornith-1.5-35B-A3B-MLX-4bit` checkpoint through the same
Swiftlet path. Official produced 2676 tokens with real JS game logic;
CRACK produced 576 tokens with broken CSS and no script tag. This resolves
the open question from `test-handoff.md` (which can now be deleted). The
conversion pipeline and Swiftlet are correct; the CRACK fine-tune's
abliteration specifically damaged long-range structural code generation.

## Also investigated this session: `--cache-gb` gap, hazard-tracking hypothesis (rejected)

Don't re-litigate this without new evidence: the "wait-minus-real-exec grows
with `--cache-gb`" observation from the original sweep (`CONVERSION_PLAN.md`
"`--cache-gb` sweep: not a lever here") was followed up with a specific
mechanism -- `ExpertCache` slot buffers are the one persistent,
manually-synchronized buffer pool in the codebase that never got
`.hazardTrackingModeUntracked` (everything else under the same
synchronization discipline in `QwenMetalModel.swift` already has it) --
implemented, verified byte-identical + 63/63 tests, then A/B-measured
across `--cache-gb` 2/4/6/8. **Rejected**: the gap grew almost identically
with or without it, slightly worse at `--cache-gb 8` across a repeat run.
Change reverted, nothing landed. Full numbers in `CONVERSION_PLAN.md`
"`--cache-gb` gap revisited: hazard-tracking hypothesis tested, rejected".
Memory pressure (18GB unified memory, not per-resource driver overhead) was
the remaining untested explanation -- see the next section, now tested.

## New this session (2026-09-03): memory-pressure diagnostic + KV-cache INT8 quantization (now the default)

**Memory-pressure diagnostic, partially confirmed.** `scratch/memory_pressure_sweep.sh`
ran `--cache-gb` 2/4/6/8 (+ repeat at 8) against `base.qpack` (`crack.qpack`
no longer exists on disk), sampling `vm_stat`/`footprint` throughout each
run. Disk swap ruled out as a factor. Memory-*compressor* churn (distinct
from swap) tracked the growing wait-minus-gpu gap across cache sizes --
real signal, not a clean 1:1 explanation. Full numbers/verdict:
`CONVERSION_PLAN.md` "Memory-pressure hypothesis: instrumented, partially
confirmed".

**KV-cache INT8 quantization: implemented, verified, now the default.**
The growing FP32 K/V cache for the 10 full-attention layers (`DecodeState.kv`)
is now quantized to INT8 by default -- group size = headDim, one affine
`scale`/`bias` pair per (position, kvHead), the same MLX-affine convention
already used for weights. `SWIFTLET_KV_QUANT=off`/`fp32` reverts to the old
FP32 path (escape hatch, same shape as `SWIFTLET_NO_MOE_BATCH`). New code:
`KVQuant.swift` (the quantized cache type), `QwenMetalModel.attnCoreInt8`
(factors the affine dequant out of the `cblas_sgemv` reduction rather than
materializing dequantized floats, mirroring `gemv_affine`'s trick).
Real-data prototyping (`scratch/kv_quant_prototype.py`, against actual
dumped K/V) found INT4 had a real ~10-20% error at per-token granularity --
ruled out; INT8 measured near-lossless (cosine similarity >0.9999), so
that's what shipped.

Verified in stages: unit tests at short context (68/68 passing, up from 63
-- two new: `kvQuantInt8ExercisesRealAttention`, `kvQuantInt8LogitsCloseToFP32`);
200-token real-model runs (FP32 path byte-identical to the pre-change
baseline when reverted via the env var; INT8 path coherent). Then, since
KV cache is too small a fraction of total footprint at 200 tokens to see
any memory effect, a real **long-context sweep** at ~20-25K tokens
(`scratch/kv_quant_long_context_sweep.sh`, FP32 vs INT8, `--cache-gb 2`,
sequential runs, `footprint` sampled throughout): **+14% decode throughput,
-20% peak process footprint, no coherence regression**. Full numbers:
`CONVERSION_PLAN.md` "Long-context sweep". One negative-but-useful finding
from that sweep: the `--cache-gb` wait/gpu gap above did *not* shrink under
INT8 at a fixed cache budget -- it's driven by expert-cache traffic, not KV
size, at least at `--cache-gb 2`.

Existing GPU-vs-CPU-oracle parity tests (`gpuMatchesCPUOnQwen35Tiny`,
`gpuMatchesCPUOnQuantizedTiny`, `gpuQpackStreamingMatchesCPU`,
`streamingInstallMatchesRepacker`) now pin `forceKVQuantMode: .off` --
they assert near-bit-exact parity against the CPU oracle (which stays
FP32 always) at a `<2e-3` tolerance calibrated for numerically-equivalent
changes, and INT8 KV quant is the first genuinely lossy change in this
whole arc, so it needed its own tests at a deliberately looser tolerance
instead of loosening those.

Not wired into `ExpertCacheMemoryGovernor`'s cache-sizing math -- confirmed
that path isn't on `swiftlet generate`'s actual route today regardless (see
`synopsis.md` "Verifying all parts of the port"), so this is a real,
deliberate scope limit, not an oversight. `ArchConfig.kvBytesPerTokenInt8`
exists for accounting/future wiring.

## MTP investigation (2026-09-03): background, what's confirmed, what's next

Speculative decoding was scoped as the next lever after router-aware expert
prefetch (see "Status as of 2026-09-03" at the top of this file). Two
drafter options exist: a training-free n-gram/prompt-lookup drafter, or this
model's own trained MTP (multi-token-prediction) head, present in the
source GGUF but dropped by this project's conversion pipeline. This section
is the background for pursuing the MTP option specifically.

### Why plain (n-gram) speculative decoding is a weak bet here

Verifying K draft positions in one batched pass only pays off if the
underlying compute cost stays close to K=1's cost (the classical
dense-model "free lunch": batching doesn't add much because you're already
memory-bandwidth-bound reading the same weights). That assumption holds for
this model's dense parts (attention, GatedDeltaNet, shared expert, router)
but **not for the routed experts** -- each of K nearby positions can pick a
different top-8 of 256 experts. Measured directly from real router traces
(`scratch/router_overlap_analysis.py` over `SWIFTLET_ROUTER_TRACE=<path>`
dumps, `QwenMetalModel.RouterTraceRecorder`, two 300-token runs, prose and
code prompts): verifying a K=8 draft window costs **~4.2-4.5x** the MoE
compute of verifying K=1, not ~1x (K=2: ~1.6x; K=4: ~2.7x). Real reuse
exists (roughly half the "fully disjoint" worst case at K=8) but nowhere
near free. A training-free n-gram drafter's natural operating point is
often K=4-8+ (long lookup matches), right where this cost is worst, and its
acceptance rate is weak outside literally-repetitive text (code, not
general prose) -- a bad combination. Not ruled out entirely, but a shaky
bet without first measuring acceptance rate on real prompts.

### Why MTP is the better bet, and what backs that up (not just naming)

The MoE-overlap cost above applies identically to *any* drafter -- it's a
verification-side cost. What changes with MTP is K and acceptance rate:

- **This checkpoint's MTP module has exactly one stage (K=1 native draft
  depth)** -- confirmed no `nextn.*` tensors exist outside block 40
  (`scratch/gguf_tensor_infos.json`, grep for `nextn` outside `blk.40`:
  zero results). K=1-2 is right where the MoE-overlap cost is mild (~1.6x
  at K=2), not the K=8 regime where it's worst.
- **MTP heads are trained jointly with the base model specifically to
  predict its own continuations** -- acceptance rate on general prose is
  typically much higher than n-gram lookup, which only works when text is
  literally repetitive.
- **The mechanism is now identified with real confidence, not just tensor
  naming.** `blk.40`'s tensors (`scratch/gguf_tensor_infos.json`, `grep
  'blk.40'`) are a *complete extra hybrid layer* (its own `attn_q/k/v/
  output` + `attn_q_norm/k_norm` + `attn_norm`/`post_attention_norm`, full
  GQA-attention-family tensors matching this model's other full-attention
  layers exactly, including the doubled `attn_q` output width from the
  query/gate fusion -- **not** a GatedDeltaNet linear-attention layer, plus
  its own routed+shared MoE: `ffn_gate/up/down_exps`, `ffn_gate_inp`
  (router), `ffn_*_shexp`) *plus* four combination tensors:
  `nextn.eh_proj.weight` (dims `[4096, 2048]`), `nextn.enorm.weight` `[2048]`,
  `nextn.hnorm.weight` `[2048]`, `nextn.shared_head_norm.weight` `[2048]`.
  `eh_proj`'s exact `4096 -> 2048` shape is the tell: `4096 = 2 x 2048`
  (this model's `hidden_size`), matching **DeepSeek-V3's published MTP
  design** tensor-for-tensor: RMSNorm the embedding of a candidate
  next-token (`enorm`) and the previous hidden state (`hnorm`) separately,
  concatenate **in that order -- `enorm`'s output first, `hnorm`'s
  second** (confirmed against vLLM's real `forward()`, see "MTP wiring
  check against a real reference" below -- this file originally had the
  order ambiguous/reversed here, a real bug this check exists to catch),
  project down through `eh_proj`, run the result through the MTP module's
  own transformer block (block 40's attention+MoE, same as above), RMSNorm
  the output (`shared_head_norm`), then reuse the **main model's own
  `lm_head`** for logits (confirmed separately: `output.weight`
  != `token_embd.weight`, `tie_word_embeddings: false` in `config.json`, so
  this checkpoint already loads a real dedicated `lm_head` -- see
  `QwenMetalModel.lmHead` -- no separate MTP output head to port).
  `scratch/build_hf_checkpoint.py`'s own header comment already independently
  named block 40 as "the MTP nextn block" during the original conversion,
  before this session looked at it again -- consistent, not a new guess.
- **Numeric sanity check: done, passed.** (Was "not yet checked" as of the
  prior session -- see "Numeric sanity check: done" below.) No comparison
  against a reference MTP implementation (DeepSeek-V3's own, or any port of
  it) has been attempted -- that's the next open item if pursuing further.

### Numeric sanity check: done, this session (2026-09-03, later still)

Step 1 above is resolved: **`enorm`/`hnorm`/`shared_head_norm`/`eh_proj` all
look like real trained weights, not garbage or uninitialized data.** Ran
without re-downloading the full 21.7GB GGUF -- since `gguf_tensor_infos.json`
already has every tensor's absolute file byte offset (`GGUFReader`'s
`data_offset` semantics: `start_offs + offset_tensor`, confirmed contiguous
across block 40's tensors), fetched only the needed byte ranges (~4.8MB
total) via HTTP `Range` requests against the HF resolve URL, which redirects
to an XET CDN URL that honors `Range` (`accept-ranges: bytes`, verified with
a `curl -I` HEAD first). Script + output:
`scratch/mtp_numeric_sanity.py`/`scratch/mtp_numeric_sanity.out`.

Results, `blk.40`'s two known-good norms (its own real `attn_norm`/
`post_attention_norm`, part of its ordinary attention layer) vs. the three
`nextn.*` norms:

| Tensor | mean | std | min | max | nan/inf |
|---|---|---|---|---|---|
| `attn_norm` (reference) | 1.0001 | 0.0030 | 0.9896 | 1.0115 | 0/0 |
| `nextn.enorm` | 0.9787 | 0.0010 | 0.9777 | 0.9910 | 0/0 |
| `nextn.hnorm` | 1.0157 | 0.0056 | 0.9819 | 1.0215 | 0/0 |
| `nextn.shared_head_norm` | 1.0228 | 0.0002 | 1.0219 | 1.0231 | 0/0 |
| `post_attention_norm` (reference) | 1.0148 | 0.0036 | 1.0056 | 1.0215 | 0/0 |

All five: mean within ~2% of 1.0, tight std, no NaN/Inf -- the `nextn.*`
norms are statistically indistinguishable in character from block 40's own
already-relied-upon norms. No sign of the "+1.0 correction" false lead
`build_hf_checkpoint.py`'s header comment warned about (that would show up
as mean near 0, not near 1). `nextn.eh_proj.weight` (Q4_K, dequantized to
`[2048, 4096]`) also looks like an ordinary trained linear-layer weight:
mean ~0, std 0.0195, bounded to roughly [-0.11, 0.10], no NaN/Inf, not
degenerate/all-zero.

**Verdict: proceed.** Nothing here contradicts the DeepSeek-V3-shape-match
hypothesis; this was the cheap check that could have killed the idea early
(e.g. if `eh_proj` had turned out to be a leftover/dead weight or the norms
looked like noise) and it didn't. The next open item is the one this
section couldn't resolve on its own -- no reference DeepSeek-V3 MTP
implementation was run for a real forward-pass comparison (numeric
sanity of static weights alone can't confirm the *wiring* -- concat order,
which hidden state feeds `hnorm` vs. `enorm`, etc. -- is right).

### `build_hf_checkpoint.py` extended to keep block 40: done, this session (code + unit test only, no full run)

Step 2 (below) is done at the code level, deliberately scoped to avoid the
21.7GB GGUF re-download + ~65GB intermediate-checkpoint regeneration the
full pipeline would need (disk is tight -- 74GB free -- and this is still
research-stage work; user chose "code + unit test, no full run" when asked).

- New `ORNITH_KEEP_MTP_BLOCK=1` env flag (off by default, same opt-in-escape-
  hatch shape as `SWIFTLET_NO_MOE_BATCH`/`SWIFTLET_KV_QUANT`), gating whether
  `blk.40` is dropped or kept.
- New `MTP_PASSTHROUGH` name mapping for the four `nextn.*` combination
  tensors, following **DeepSeek-V3's own published HF checkpoint
  convention** (not invented): the MTP module keeps the same
  `model.layers.{N}.` prefix as an ordinary decoder layer, so
  `nextn.eh_proj`/`enorm`/`hnorm`/`shared_head_norm` become
  `model.layers.40.eh_proj`/`enorm`/`hnorm`/`shared_head.norm`. Block 40's
  own attention+MoE tensors need **no new mapping at all** -- confirmed
  identical dims/`ggml_type` to `blk.39` (an ordinary, already-verified
  full-attention-MoE layer) for every shared suffix, so they resolve
  through the existing `PASSTHROUGH` dict unchanged.
- Fixed a latent bug while in there: `target_key()` was dead code (never
  called -- `main()` duplicated its logic inline, and the two had actually
  drifted: `target_key` silently returned `None` for an unmapped top-level
  tensor where `main()` raised `ValueError`). Refactored so `main()` now
  calls `target_key()` as the single source of truth; also gave it the
  "fail loud on genuinely unmapped, `None` only ever means dropped MTP
  block" contract explicitly.
- **Verified, no full pipeline run needed**: new
  `scratch/test_mtp_block_mapping.py` fetches block 40 (confirmed
  contiguous on disk, ~521MB) in a single HTTP Range GET against the HF
  resolve URL -- no local GGUF, no download of the other ~21GB -- then for
  all 20 real block-40 tensors checks `target_key()`'s mapping under both
  flag states plus the full dequant/shape pipeline (shapes match, no
  NaN/Inf, RMSNorm gains mean near 1.0). All pass. Separately verified
  **733/733** non-MTP tensor names across the whole model still resolve to
  byte-identical keys with the flag unset -- the default pipeline is
  unaffected, not just "should be."
- **Explicitly still unverified** (would need the full run): whether
  `mlx_lm.convert`'s actual loader tolerates the resulting extra
  `model.layers.40.*` keys given `config.json`'s `num_hidden_layers` still
  says 40 -- this was flagged rather than guessed at, see the script's
  updated docstring and `MTP_PASSTHROUGH`'s comment.

### MTP wiring check against a real reference: done, this session

Resolves the remaining half of step 1: found that neither DeepSeek-V3's own
official inference repo (`deepseek-ai/DeepSeek-V3`, `inference/model.py`)
nor HF `transformers`' `deepseek_v3` modeling code implement the MTP module
at all -- both drop it, same as this project's own `build_hf_checkpoint.py`
did before this investigation. **vLLM does** (it's the one place MTP is
actually used, for its own speculative decoding):
`vllm/model_executor/models/deepseek_mtp.py`
(`DeepSeekMultiTokenPredictorLayer.forward`, fetched at
`github.com/vllm-project/vllm@88b2bff2c63d0f28396451f1199d09ee0f3e2d88`,
2026-08-18) is a real, maintained reference for exactly this module.

Used **synthetic** inputs (random hidden state + random stand-in token
embedding) rather than a real forward pass through the 35B model -- the
question is whether the *wiring* matches, which a random input answers
identically to a real in-context one, without needing the full model or
GGUF at all. Used the REAL `enorm`/`hnorm`/`shared_head_norm` weights
(fetched the same way as the numeric sanity check); `eh_proj` used a random
weight of the correct shape (a wiring-order bug shows up identically either
way -- see result). `mtp_block` (block 40's own attention+MoE) was stubbed
identically on both sides -- that's this model's ordinary, already-verified
decoder layer, not what's in question here; this check isolates the
combination step around it. Script: `scratch/mtp_wiring_check.py`.

**Real finding, not just confirmation: caught a genuine ambiguity in this
file's own prior prose.** This file's earlier description --
"RMSNorm the previous hidden state (`hnorm`) and the embedding of a
candidate next-token (`enorm`) separately, concatenate" -- lists `hnorm`
before `enorm`, and a literal reading concatenates them in that order.
vLLM's actual `forward()` does it the other way: `torch.cat([inputs_embeds,
previous_hidden_states])` -- **`enorm`'s output (the token embedding half)
goes first, `hnorm`'s (the hidden-state half) second** into `eh_proj`'s
`[hidden, 2*hidden]` weight. Feeding both orderings through the identical
weights and comparing: **max abs diff 5.13** on logits of scale ~1 --
nowhere close to a rounding difference, a real silent-wrong-answer bug if
ported the way this file's prose would have naturally been read. Corrected
in the architecture description above; **the concat order for the Swift
port is confirmed: `[enorm(token_embed), hnorm(prev_hidden)]`.**

Also newly noted from reading vLLM's loader code (`_rewrite_spec_layer_name`
+ the MTP-completeness check in `load_weights`): real DeepSeek-V3
checkpoints ship a **separate `shared_head.head` weight** for the MTP
module (vLLM allocates its own `ParallelLMHead` per MTP layer, not
automatically tied to the main `lm_head`). This project's own GGUF has no
such tensor in `blk.40` (confirmed: the 20-tensor list in the mapping-check
section above is exhaustive, no `nextn.shared_head`-weight or similar) --
so either this specific checkpoint's conversion omitted a
byte-identical-to-`output.weight` duplicate (plausible, common
space-saving move), or it never had one. Flagged, not resolved: low risk
either way, since getting this wrong only degrades MTP's *draft* quality
(observable via a bad acceptance rate under the project's own
rejection-sampling correctness bar in step 6 below), not final-output
correctness -- appropriate to resolve empirically once there's a real
forward pass to test acceptance rate against, not by more research now.

### Swift/Metal port (step 3): done, this session -- combination step + real attnForward/moeForward reuse, verified against a real mlx-lm fixture

Repo: `Swiftlet/` (uncommitted working-tree changes, same convention as
everything else -- ask before committing). Chose the fuller of two scoped
options (full reuse path + synthetic test, vs. combination-step-only with
a stubbed block-40 forward) when asked -- see below for why the "reuse"
half wasn't quite as free as this file's own prior wording implied.

- **New `QwenMetalModel.MTPGPU` struct** (`enorm`/`hnorm`/`sharedHeadNorm`
  + `ehProj: GPULinear` + `block: LayerGPU`) and `mtpLayer: MTPGPU?`
  property, loaded in `init` gated purely on
  `ckpt.contains("model.layers.\(numHiddenLayers).enorm.weight")` -- nil
  or absent-tensor on every checkpoint this project has actually built so
  far, zero effect on any existing model directory. Deliberately
  duplicates (rather than refactors to share) the main loop's
  full-attention-layer-loading branch -- that loop is the verified default
  path for every real run; this new, as-yet-unexercised-on-real-data
  branch shouldn't touch it.
- **One real bug caught by actually reading `QwenCPUModel.DecodeState`
  before assuming "straight reuse" was free**: `DecodeState.kv`/`kvQuant`
  turned out to be `[Int: ...]`-keyed dictionaries, not fixed
  `0..<numHiddenLayers` arrays -- so `attnForward`/`moeForward` accept
  `layerIndex: config.numHiddenLayers` (one past the real stack) with
  zero new state-management plumbing, exactly as hoped. The one place the
  reuse ISN'T free: `moeForward`'s expert-cache path needs a qpack
  manifest section for that layer index, which doesn't exist yet (no
  full-pipeline run) -- worked around by using Swiftlet's existing
  non-qpack "stacks" fallback path instead (`ExpertStack`, already used by
  every non-cached model directory), which needs no qpack support at all.
- **New `mtpDraftForward(candidateTokenEmbed:previousHiddenState:state:)`**
  (`QwenMetalModel.swift`): `enorm`/`hnorm` RMSNorm, concat in the
  vLLM-confirmed `[enorm, hnorm]` order, `eh_proj` GEMV, then a straight
  reuse of `attnForward`+`moeForward` (the exact per-layer body
  `stepOne`'s main loop already uses: rmsNorm -> attn -> residual add ->
  rmsNorm -> moe -> residual add) at `layerIndex: numHiddenLayers`,
  finished with `sharedHeadNorm`. Also fixed a real latent buffer-overflow
  risk while wiring this in: `xBuf`'s size (`maxIn`) didn't account for
  `eh_proj`'s `2*hiddenSize`-wide input, and `loadX`'s `copyMemory` has no
  bounds check -- would have been a real memory-corruption bug the first
  time this path actually ran, caught by reading `loadX` before calling
  it, not by a crash.
- **New fixture, `scripts/gen_mtp_fixture.py`**: extends
  `gen_fixtures.py`'s existing tiny-qwen3_5 generator (same ARGS/seed as
  `tiny-model-q35`) with one extra real `qwen3_5.DecoderLayer` (forced
  full-attention via `layer_idx=3` under `full_attention_interval=4` --
  `layer_idx` only selects attention-vs-DeltaNet internally, not
  serialized) plus the four MTP combination modules, at layer index 8 (one
  past the tiny model's real 8-layer stack) -- exactly how block 40 sits
  relative to this project's real 40-layer stack. Computes the reference
  forward with the same vLLM-confirmed wiring
  (`scratch/mtp_wiring_check.py`'s order), using a real in-context hidden
  state (the tiny model's own last-layer output over `gen_fixtures.py`'s
  TOKENS prompt) rather than a fully synthetic one. Output:
  `fixtures/tiny-model-q35-mtp/` + `fixtures/tiny_forward_q35_mtp
  .safetensors`/`.json`.
- **New tests, `MetalModelTests.swift`**: `mtpDraftForwardMatchesMLX`
  (real `QwenMetalModel` against the fixture above, `<2e-3` tolerance --
  same convention as every other GPU-vs-reference test in this suite) and
  `mtpDraftForwardConcatOrderMatters` (a test-only
  `swapConcatOrderForTesting` parameter on `mtpDraftForward`, same
  test-only-parameter shape as `forceKVQuantMode` etc., proves the fixture
  actually has discriminating power over concat order rather than passing
  vacuously). **All 77 Swiftlet tests pass** (was 68 as of the previous
  MTP-investigation update in this file; the gap is unrelated
  already-in-flight work -- `ExpertCacheTests.swift`, prefetch tests --
  not from this change).
- **Explicitly still not done**: not wired into any real decode loop
  (that's step 5, the speculative-decode loop itself, a separate
  concern from porting the module); not yet exercised against real
  (non-synthetic-fixture) weights, since no checkpoint with an actual MTP
  block has been built end-to-end (still needs step 2's deferred full
  pipeline run).
- **Done, background**: the 21.7GB GGUF finished downloading to
  `scratch/downloads/Ornith-1.5-35B-A3B-CRACK-Q4_K_M.gguf` (user asked for
  this alongside the Swift work, to have it ready for the eventual full
  pipeline run). Size verified exact (21,713,462,848 bytes, matches the HF
  CDN's `x-linked-size` header from earlier this session). Disk: 55GB free
  after the download -- enough for the GGUF alone but NOT also the ~65GB
  intermediate HF checkpoint the full pipeline would need next; that step
  still needs the same delete-the-GGUF-once-dequantized staging
  synopsis.md documents. `build_hf_checkpoint.py`'s `GGUF_PATH` already
  points at this exact path, so the full pipeline run (step 2's remaining
  half) is unblocked and ready whenever wanted -- not run yet, not asked
  for this session.
- **Checked, not enough free space to run the full pipeline yet**: 56GB
  free (`df -h`) after the GGUF download, vs. the ~65-67GB the
  dequantization step needs for its output HF checkpoint (block 40 pushes
  it slightly above the ~65GB baseline) -- a ~10GB shortfall, and it can't
  be closed by deleting the GGUF first (`build_hf_checkpoint.py` reads
  from it throughout that step). Looked for reclaimable space in
  `scratch/`: nothing else is meaningfully large except
  `scratch/ornith-1.5-35b-base.qpack` (18GB) -- but that's the currently
  active, working model this whole project benchmarks against (see
  "How to reproduce" above), not disposable scratch, so it was NOT deleted
  without asking. Options if/when resuming: free space elsewhere on the
  machine (outside this project), delete `base.qpack` (loses the working
  model until rebuilt -- needs explicit go-ahead), or accept running the
  dequant step close to the edge.

### Full pipeline run + `mlx_lm.convert` blocker found and worked around: done (2026-09-04)

Disk was freed (108GB avail at the start of this sub-session, well over the
~87GB peak need); picked up exactly at step 2's deferred half.

**Full `build_hf_checkpoint.py` run, `ORNITH_KEEP_MTP_BLOCK=1`: done.** 753/753
GGUF tensors written, **0 MTP tensors skipped** (vs. 733 written/20 skipped
for the text-only baseline) -- `scratch/hf-checkpoint/` (66GB), all 20
`model.layers.40.*` keys present and correctly named per `MTP_PASSTHROUGH`.
Confirmed `config.json`'s `num_hidden_layers` is still 40 (as flagged), so
layer 40 is genuinely "extra" relative to the declared stack -- this is
exactly the condition the prior session flagged as untested.

**`mlx_lm.convert` on that checkpoint: fails, hard blocker, root cause fully
traced (not guessed).** `mlx_lm/utils.py`'s `load_model()` calls
`model.load_weights(list(weights.items()), strict=True)` with no CLI-exposed
way to pass `strict=False` through `convert()`; since mlx-lm's
`Qwen3_5TextModel` instantiates exactly `num_hidden_layers` (40)
`DecoderLayer`s, every one of block 40's 20 keys is "extra":

```
ValueError: Received 20 parameters not in model:
language_model.model.layers.40.eh_proj.weight, ...
```

**Bumping `num_hidden_layers` to 41 would not have been a safe workaround --
checked, not just assumed.** `DecoderLayer.is_linear = (layer_idx + 1) %
full_attention_interval != 0` (`qwen3_5.py`): at `layer_idx=40` that's `41 %
4 = 1 != 0`, so mlx-lm would build layer 40 as **GatedDeltaNet
(linear-attention)**, not full self-attention -- wrong architecture for a
block whose real tensors (confirmed shape-for-shape against the actual
converted checkpoint below) are full GQA self-attention, matching blk.39.
The 4 `nextn.*` combination tensors (`eh_proj`/`enorm`/`hnorm`/
`shared_head.norm`) would *also* still have no matching module regardless of
layer count, since `DecoderLayer` has no such submodules -- same problem
DeepSeek-V3's own HF/vLLM code has, per the wiring-check section above.

**Resolution, user-chosen (offered as one of three options: hand-quantize,
patch the installed `mlx_lm` dependency, or stop here): hand-quantize block
40 directly, bypassing mlx-lm's Model class entirely for it.** This matches
how the Swift port already treats block 40 (`mtpLayer: MTPGPU?`, a
deliberately separate, hand-loaded property, not part of the main decoder
stack) -- Swiftlet's own loader reads tensors by name, never through
mlx-lm's Python `Model`, so mlx-lm never needs to accept these keys.

Two new scripts, both in `scratch/`:
- **`make_base_view.py`**: builds a disk-cheap "view" of `hf-checkpoint/`
  for a normal 40-layer conversion -- symlinks every shard except the one
  holding layer 40's tensors (zero extra disk), rewrites that one shard
  without the 20 layer-40 keys. `mlx_lm.convert` against this view
  succeeded and reproduced the *exact* known-good baseline (**4.503 bits
  per weight**, byte-identical to `mlx_convert3.log`'s prior run) --
  confirms nothing else regressed.
- **`quantize_merge_mtp.py`**: quantizes block 40's 20 tensors by hand with
  `mx.quantize` and merges them into the now-converted `mlx-checkpoint/` as
  a new shard (`model-mtp-block40.safetensors`) plus an updated
  `model.safetensors.index.json`. The quantization convention was
  **reverse-engineered from mlx-lm's own real output, not guessed**:
  inspected the actual `mlx-checkpoint/config.json`'s per-path
  `quantization` overrides and found `qwen3_5.py`'s
  `Qwen3_5TextModel.quant_predicate` -- `group_size=64`/`mode=affine`
  throughout (this project's standard CLI args); **8-bit** for paths ending
  `mlp.gate` or `shared_expert_gate` (the router and shared-expert gate);
  **4-bit** for every other Linear-family tensor (`self_attn.{q,k,v,o}_proj`,
  `mlp.shared_expert.{gate,up,down}_proj`, `mlp.switch_mlp.{gate,up,down}_proj`,
  `eh_proj`); the 7 RMSNorm vectors stay bf16, unquantized (`nn.RMSNorm` has
  no `to_quantized`, confirmed against the real checkpoint's own norms).
  `SwitchLinear.to_quantized`'s source confirms `mx.quantize` applies
  directly to the raw per-expert 3D weight array, no special-casing needed
  for `switch_mlp`'s batched tensors.
- **Verified**: unquantized norms are byte-exact (max abs diff 0, all 7)
  after the merge; the 13 quantized tensors dequantize back with ~99.6%
  cosine similarity / ~8.9% relative error at 4-bit (**matches the rest of
  this model, which already ships at this same group-64/4-bit scheme** --
  expected lossiness, not a bug) and ~99.997% at 8-bit for the two
  router/gate paths (much tighter, consistent with why mlx-lm picks 8-bit
  there specifically).

**Result**: `scratch/mlx-checkpoint/` (19GB) is now the **first MTP-inclusive
quantized MLX checkpoint this project has produced** -- 1803 tensors total,
block 40's 20 fully present and quantized to the same scheme as the other 40
layers. `scratch/hf-checkpoint/` (66GB, MTP-inclusive dequantized
intermediate) is still on disk, regenerable via `ORNITH_KEEP_MTP_BLOCK=1
python3 build_hf_checkpoint.py` against the already-downloaded GGUF --
flagged as a deletion candidate to free space (same
not-deleted-without-asking discipline as `base.qpack`), not removed this
session. Disk: ~20GB free after this work (`hf-checkpoint-base`, the
disk-cheap symlinked view, was cleaned up since `make_base_view.py`
regenerates it in seconds).

### `swiftlet-repack`: real gap found and fixed, verified end-to-end (2026-09-04)

Ran `swiftlet-repack --source mlx-checkpoint --output mtp-test.qpack` against
the real MTP-inclusive checkpoint above. **It "succeeded" -- no error -- but
silently dropped block 40's routed-expert weights.** Traced why by reading
`QpackRepacker.repack()` (`Qpack.swift`), not guessed:

- The per-layer expert-stack packing loop is hardcoded `for layer in
  0..<config.numHiddenLayers` (0..39) -- block 40 was simply never reached,
  no `layer_40.bin` was written.
- The dense-tensor sweep (`for name in ckpt.tensorNames.sorted()`) has no
  layer-count bound at all -- it excludes anything matching
  `.mlp.switch_mlp.` or containing `mtp.`/`.mtp.`, and since this project's
  tensor names follow DeepSeek-V3's own convention (`eh_proj`, `enorm`, ...,
  no literal "mtp" substring), **37 of block 40's 46 real tensors landed in
  the qpack's dense `model.safetensors` file by accident** (confirmed by
  listing them directly) -- only the 9 `switch_mlp.{gate,up,down}_proj`
  weight/scales/biases keys were missing, exactly the ones the per-layer
  loop was supposed to produce.
- Checked whether this would actually bite at runtime, not just in the
  packed bytes: `moeForward`'s qpack-mode routed-expert fetch
  (`QwenMetalModel.swift`) calls `expertCache.buffers(layer: layerIndex,
  ...)` with `layerIndex = cfg.numHiddenLayers` (40) for MTP's draft call;
  `ExpertCache`/`QpackExpertReader` size their `fds` array from
  `layout.layerCount` (40, unfixed) -- so the first real MTP draft
  invocation against an unfixed qpack would index out of bounds. Real,
  confirmed failure mode, not theoretical.

**Fixed in `Qpack.swift`**: `QpackRepacker.repack()` now detects the MTP
block with the identical presence check `QwenMetalModel.swift`'s `mtpLayer`
already uses (`ckpt.contains("model.layers.\(numHiddenLayers).enorm.weight")`),
and when present, extends `layerCount` to `numHiddenLayers + 1` and the
per-layer packing loop to match, plus appends one `linearLayers` entry
(hardcoded `false` -- block 40 is confirmed full-attention, not derived from
the periodic `isLinearLayer` formula, mirroring the identical reasoning
already in `QwenMetalModel.swift`'s `mtpLayer` construction). Zero-effect
on every checkpoint without the block (confirmed: all 77 tests still pass,
including `repackRoundTrip`, `officialConfigIsAccepted`,
`explicitProfileMatchesOrnithGeometry`, `streamingInstallMatchesRepacker`,
and the stricter `OrnithSupportTests`/`OrnithRuntimeFactory` production-path
validators that hardcode `ArchConfig.ornith1_5_35B.layerCount == 40` --
deliberately untouched, since those gate the *official* non-MTP distribution
path via `swiftlet-server`, a separate concern from this project's own
local `--source` repack path that `swiftlet generate` actually uses).

**Verified end-to-end, empirically, not just by re-reading the source**:
re-ran repack against the real `mlx-checkpoint/` -- `layer_40.bin` now
exists (432 MiB, same size as every other layer, as expected since block
40's routed-expert structure mirrors blk.39's exactly); `layout.json` shows
`layerCount: 41`, 41 `linearLayers` entries ending `[..., false, false]`
(layers 39 and 40 both full-attention, matching the interval-4 pattern).
Then an actual **smoke-test `swiftlet generate` run against the extended
qpack** (`--gpu --chat --cache-gb 2`, real prompt): model loaded
successfully (`ExpertCache` initialized fine against the now-41-layer
layout, no crash), and ordinary decoding produced normal, coherent output
at a normal decode rate (8.09 tok/s this run -- MTP isn't wired into the
decode loop yet, so this exercises only that loading/init didn't break, not
speculative decoding itself). Cleaned up the throwaway `mtp-test.qpack`
afterward (19GB, regenerable via the repack command above in ~70s) --
disk was briefly down to 4.5GB free during this.

### `mtpDraftForward` exercised against real weights for the first time: done, a real bug found and fixed (2026-09-04)

Added `QwenMetalModel.mtpSanityCheck(realLogits:state:)`, opt-in via
`SWIFTLET_MTP_CHECK=1` (same flag shape as `SWIFTLET_CATEGORY_TIMING`),
called once from `runGenerate` right after a real prefill. Takes the
model's own greedy pick from the just-computed real logits as a stand-in
"candidate token" and the real pre-final-norm hidden state (read from
`hBuf`, confirmed correct by reading `step()`: it calls `stepOne` once per
token, so `hBuf` holds exactly the last logits-producing token's hidden
state), feeds both through `mtpDraftForward`, projects the result through
the shared `lmHead`, and reports whether the output is numerically sane.

**First run crashed -- a real bug, not a synthetic edge case.** `Index out
of range` inside `attnCoreInt8` (INT8 KV-quant is the default). Root cause,
confirmed by reading the code: `attnCoreInt8` computes `kvLen = state.position
+ 1`, assuming every layer's quantized KV cache has grown in lockstep with
the shared global position -- true for the main 40 layers (called every
real step) but false for layer 40, whose `state.kvQuant[40]` entry this
very call creates fresh (one position's worth after `qcache.append`), while
`kvLen` still claimed `state.position + 1` (22, after a 21-token prefill) --
reading `kScale`/`kBias` far past their actual one-entry length.

**Real design implication for step 5** (the speculative-decode loop, not
yet built): the MTP module's dedicated attention weights mean its own KV
cache needs to be populated at every real step it must attend back over,
not just conjured at the moment of drafting -- a genuine piece of the
speculative-decode design this crash surfaced, not something the sanity
check itself needed to solve. Worked around *for this diagnostic only* by
giving `mtpSanityCheck` its own fresh `DecodeState` (position 0, so
`kvLen=1` matches the freshly-appended cache by construction) --
documented in the code as a real, known limitation: this exercises real
weights and the module's numeric behavior, but not real full-context
attention. That remains open for step 5 to solve properly.

**Result, after the fix, real weights, real qpack (rebuilt via the fixed
`swiftlet-repack`)**: no crash, no NaN/Inf.
```
mtpSanityCheck: candidate=760 (real argmax) -> draft top=1156 (logit 4.974),
range=[-6.470, 4.974] mean=-1.9936 nan=no inf=no
```
Plausible logit magnitudes for this model's 248,320-token vocabulary (not
degenerate/collapsed/exploded); the draft's top pick differs from the
"candidate" token fed in, as expected (the draft predicts the *next*
position given a hypothetical token at this one, not an echo). Verified:
all 77 tests still pass after both the crash fix and the original hook.
This is the first real, non-synthetic, non-static-weight signal that the
ported module runs correctly end-to-end (GGUF -> hand-quantized MLX
checkpoint -> fixed qpack repack -> Swift/Metal port) -- distinct from both
the static-weight check (`mtp_numeric_sanity.py`) and the tiny-fixture
wiring check (`mtpDraftForwardMatchesMLX`) done earlier.

Cleanup: the verification qpacks built during this and the repack-fix work
(`mtp-test.qpack`, `mtp-check.qpack`, ~19GB each) were deleted after use --
regenerable from `mlx-checkpoint/` via `swiftlet-repack` in under a minute.
`scratch/hf-checkpoint/` (the 66GB dequantized intermediate) was also
deleted -- regenerable from the GGUF (`scratch/downloads/`, kept) via
`ORNITH_KEEP_MTP_BLOCK=1 python3 build_hf_checkpoint.py` in ~2.5 min. Kept:
the GGUF, `scratch/mlx-checkpoint/` (19GB, the real deliverable this whole
arc produced), and `scratch/mtp_check_run2.err` (the run log with the
result above). Disk: ~89GB free.

**Explicitly still open**: the `kvLen`/lockstep-KV-population design
question above is the main new piece of real uncertainty step 5 needs to
resolve (bigger than previously scoped -- it's not just "checkpoint/restore
GatedDeltaNet state", item 4 below, but also "run the MTP module's own
attention at every step, not just when drafting"). Acceptance rate is still
completely unmeasured -- nothing short of a real speculative-decode loop
answers that.

### If pursuing further: concrete next steps, roughly in order

1. ~~Numeric sanity check~~ / ~~wiring check against a reference~~ -- both
   done above. No further verification is planned before the Swift port;
   remaining uncertainty (acceptance rate, the `shared_head.head` question)
   needs a real forward pass to resolve, not more research.
2. ~~Extend `scratch/build_hf_checkpoint.py`~~ / ~~full pipeline run~~ --
   both done above, including working around the `mlx_lm.convert` strict-load
   blocker. `scratch/mlx-checkpoint/` is a real, verified, MTP-inclusive
   quantized checkpoint.
3. ~~Port to Swift/Metal~~ / ~~run `swiftlet-repack` against real
   `mlx-checkpoint/`~~ / ~~exercise `mtpDraftForward` against real weights~~
   -- all done above, including finding and fixing two real bugs (routed
   experts silently dropped by the repack tool; an INT8-KV-cache
   `kvLen`/lockstep-position crash in `attnCoreInt8`) and getting a first
   real, non-crashing, numerically-sane signal from the ported module
   against real trained weights.
4. **GatedDeltaNet state checkpoint/restore** (the originally-identified
   blocker, now unblocked by prior art -- see the "how hard is speculative
   decoding" discussion this session): small (63 MiB total), one
   `MTLBlitCommandEncoder` copy per DeltaNet layer's `FastLayer.hist`/
   `.state` before a speculative window, restore on rejection. KV-cache
   rollback (the 10 full-attention layers) is simpler -- position-indexed,
   truncate on reject, care needed for the INT8 quant metadata per
   position. **New, from item 3's crash above**: this item was previously
   scoped as "checkpoint and roll back on reject" only, implicitly assuming
   layer 40's own KV cache already exists to roll back -- it doesn't yet.
   The real prerequisite is populating layer 40's KV cache (its own
   dedicated `self_attn.{q,k,v,o}_proj` weights, not shared with the main
   stack) at every real decode step the speculative window needs to attend
   back over, not just at the moment of drafting. Scope this properly
   before starting item 5, not while already mid-implementation of it.
5. **The actual speculative decode loop + batched verification kernel**:
   real new control flow above `stepOneFast` (draft with MTP -> verify K=1
   or K=2 positions in one batched pass -> accept/reject -> rollback if
   needed), and the `gemv_moe_batched` kernel needs to grow from "K experts
   for one position" to "K experts x N draft positions." This is still the
   biggest single piece of new Metal work regardless of which drafter is
   used.
6. **Correctness**: this project's standing bar (byte-identical or
   measurably-bounded output vs. non-speculative decoding) applies here
   too, and is *harder* to hit than for prior changes in this project --
   accepted tokens must exactly match what standard decoding would have
   produced (exact-match for greedy, rejection-sampling for sampled
   decoding), and needs to interact correctly with the existing stateful
   repetition-guard/n-gram-ban sampling logic. Budget real time for this,
   not just the mechanism above.

Rough effort, revised from a first pass that didn't have the tensor-shape
confirmation above: still medium-high (the batched-verification kernel and
correctness work don't get any easier), but the *architecture-identification*
risk that made this feel research-risky rather than just engineering-risky
is now much lower given the DeepSeek-V3 shape match. The real remaining
uncertainty is acceptance rate, which step 1 above doesn't yet answer and
nothing short of an actual forward pass through the ported module will.
