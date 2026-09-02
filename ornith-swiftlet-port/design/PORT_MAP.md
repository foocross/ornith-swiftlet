# Port map

## Architectural conclusion

Ornith 1.5 35B-A3B and Swiftlet's existing Qwen3.6 35B target share the text
runtime geometry that matters here: Qwen3.5-family split DeltaNet, 40 layers,
interval-4 full attention, hidden size 2048, 256 experts, top-8 routing, and
512-wide routed/shared experts.

Accordingly, the baseline port should not fork the model graph or Metal kernels.
It should make checkpoint acceptance explicit, preserve native K=8 semantics,
and add a total-memory control plane around the existing expert cache.

## Existing Swiftlet components retained unchanged

| Concern | Current Swiftlet component | Port action |
| --- | --- | --- |
| Config parsing | `QwenConfig.swift` | Retain; add strict Ornith validation |
| Split DeltaNet | `QwenMetalModel.swift` and kernels | Retain |
| Gated GQA | `QwenMetalModel.swift` and kernels | Retain |
| Router and shared expert | Existing MoE path | Retain native top-8 |
| Expert container | `Qpack.swift` | Retain fixed-stride, one-pread blobs |
| Cache policy | `ExpertCache.swift` | Retain bounded global LFU+recency baseline |
| Tokenizer/chat/server | Existing Swiftlet products | Retain |

## Files added by the overlay

### `OrnithSupport.swift`

- Adds `ArchConfig.ornith1_5_35B`.
- Validates every field currently consumed by Swiftlet.
- Rejects accidental K changes, wrong expert counts, wrong DeltaNet layout,
  missing top-k normalization, and other silent architecture drift.

### `SlotStreamMemoryGovernor.swift`

Implements this ordering:

```text
hard target memory
  - qpack dense model.safetensors bytes
  - FP32 DeltaNet recurrence matrices and convolution tails
  - planned-context current Swiftlet FP32 KV bytes
  - scratch reserve
  - safety reserve
  = expert-cache bytes rounded down to whole qpack slots
```

This is a planning layer. Swiftlet's cache still owns residency and eviction.

### `OrnithRuntimeFactory.swift`

- Requires a qpack rather than a raw fully mapped checkpoint.
- Parses and validates the Ornith config.
- Reads qpack `layout.json` and `manifest.json`.
- Computes the hard-budget plan from actual file/layout sizes.
- Passes `plan.cacheBudgetGiB` into the existing `QwenMetalModel` initializer.

## Baseline data path

```text
Ornith MLX INT4 checkpoint
        |
        v
Swiftlet QpackRepacker
        |
        +-- model.safetensors: resident trunk/shared experts/router
        |
        +-- packed_experts/layer_XX.bin
                 |
                 v
       global bounded ExpertCache
       fixed shared Metal slots
                 |
                 v
       current Swiftlet Metal MoE path
```

## Phase-two data-path changes

Only implement these after trace evidence identifies the corresponding limit.

### Bounded parallel `pread`

Current cache miss fills are serial. A safe next design is a small fixed worker
pool that writes misses directly into already-reserved shared Metal slots. It
must deduplicate in-flight `(layer, expert)` keys, protect all members of the
current top-8 batch, and expose a barrier before GPU use.

### Persistent/fused top-8 MoE kernel

The target is fewer command dispatches, not a K reduction. Preserve the router
selection and selected-mass normalization, and compare output against the
existing scalar/fast kernels before accepting speed results.

### Prefill

The existing multi-token path still advances token by token while eliding
intermediate LM-head projections. True prefill work should batch dense and MoE
operations by chunk, with chunk size chosen from scratch headroom. Keep its
memory policy separate from decode.

### MTP

The checkpoint declares one MTP hidden layer. Current qpack deliberately omits
MTP tensors, so supporting it requires a container-version decision, loading
code, draft state, and batched target verification. It is not a config-only
change.
