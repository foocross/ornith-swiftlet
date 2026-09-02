# Handoff: Ornith-1.5-35B decode-throughput optimization

Written for the next session/model picking this up. Read `synopsis.md` first
(what this project is, how the GGUF got onto Swiftlet at all), then
`CONVERSION_PLAN.md` "Decode throughput" section (the full numbers this
document summarizes), then this file for exactly where things stand and what
to do next.

## Where things stand right now

**Decode throughput: 5.79 -> ~9.4-9.94 tok/s from the fill/eviction arc, plus
a further +5-6% from MoE kernel fusion on top of that (`--cache-gb 2`, same
reference qpack/prompt throughout).** Run-to-run variance on this machine is
real (thermal/load noise) -- treat anything in that range as "current
state," not a single precise number. All changes verified byte-identical
generated output against the pre-change baseline at every step; all 63
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

## Repo/commit state -- nothing pushed anywhere

Two separate local git repos, both untouched on origin:

- **`Swiftlet/`** (clone of `github.com/leonickson1/Swiftlet`, origin
  `main` at `aaa910a`): branch **`ornith-decode-throughput`**, four commits
  on top:
  1. `6d14f41` -- applies the pre-existing `ornith-swiftlet-port` overlay
     (was sitting as uncommitted working-tree changes before this session;
     unrelated to the throughput work, committed first to get a clean base)
  2. `62c30cc` -- bounded concurrent expert-cache fill (the big win, +56%)
  3. `491d437` -- fixed a measurement bug (prefill's expert-cache stats had
     been folded into decode's) + heap-based LFU eviction (+7% more)
  4. `14df4ea` -- opt-in per-category GPU timing diagnostic
     (`SWIFTLET_CATEGORY_TIMING=1`), off by default
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
