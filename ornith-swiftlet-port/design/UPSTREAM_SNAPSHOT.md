# Upstream API snapshot

The overlay was written against Swiftlet `main` as reviewed on 2026-09-02,
with GitHub showing commit `aaa910a` (`Add aggregate Metal step instrumentation`).
It intentionally adds files instead of copying or editing the model runtime.

The integration depends on these Swiftlet interfaces:

| File | Dependency used by the overlay |
| --- | --- |
| `ArchConfig.swift` | Qwen3.5 family enum, geometry fields, derived expert counts, and current FP16 architectural KV estimate |
| `QwenConfig.swift` | nested `text_config`, `language_model.` prefix, split DeltaNet layout, normalized top-k configuration |
| `Qpack.swift` | v1 manifest/layout Codable types and fixed-stride one-`pread` expert records |
| `QwenMetalModel.swift` | `init(modelDir:cacheBudgetGB:)` and the qpack-backed Metal runtime |
| `ExpertCache.swift` | global LFU-plus-recency cache, lazy shared-buffer slots, and its 16-slot minimum |
| `QwenCPUModel.swift` | current per-session K/V representation as Swift `[Float]` arrays |

The memory overlay deliberately distinguishes two KV numbers:

- `ArchConfig.kvBytesPerToken`: 20,480 bytes, the FP16 architectural target;
- `ArchConfig.swiftletDecodeKVBytesPerToken`: 40,960 bytes, the current runtime
  allocation used by the hard-budget planner.

It also adds the 2,949,120-byte GatedDelta convolution history that is separate
from Swiftlet's existing 62,914,560-byte recurrence-matrix property.

Before rebasing onto a later Swiftlet revision, rerun:

```sh
scripts/overlay-smoke-test.sh
git apply --check swiftlet-overlay/patches/0001-ornith-port-overlay.patch
git apply --check swiftlet-overlay/patches/0002-register-info-alias.patch
```

Then run the complete upstream test suite on Apple Silicon. The local smoke
harness guards the public API shape and deterministic planning logic; it does
not validate Metal compilation or numerical inference.
