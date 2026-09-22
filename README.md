# ornith-swiftlet

**An experiment:** how well can [Swiftlet](https://github.com/leonickson1/Swiftlet)
be ported to run `Ornith-1.5-35B-A3B`, a 35B-parameter mixture-of-experts
model (Qwen3.5-MoE architecture, ~3B active parameters per token), on a
single consumer Apple Silicon laptop under a tight memory budget?

The short answer: it works. On an M3 Pro with 18 GB of unified memory, the
model generates coherent output at **~11 tok/s**, keeping **under 4 GB**
resident at the default expert-cache size.

## Credit where it's due

This project is a **port and customization of Swiftlet, not a new inference
engine.** The heavy lifting is Swiftlet's: the Swift/Metal runtime, the
`.qpack` container and `swiftlet-repack` tool, the streaming expert cache,
and native support for the Qwen3.5 hybrid GatedDeltaNet + MoE architecture.
All of that belongs to the Swiftlet authors, and none of this would exist
without their work. Please star and cite
[leonickson1/Swiftlet](https://github.com/leonickson1/Swiftlet), not this repo.

What this repo adds on top:

- A GGUF → HF → MLX → qpack conversion path, so a llama.cpp-format Ornith
  checkpoint can be loaded by Swiftlet at all (including fixes for two
  silent tensor-layout mismatches between llama.cpp and mlx-lm).
- A small overlay of Ornith-specific patches for Swiftlet: architecture
  guards, a memory governor, and expert-cache sizing.
- A series of measured decode-throughput and memory optimizations, each
  A/B tested against the real model.

Other projects whose work informed the design, or that the pipeline relies
on: [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT),
[SlotStream](https://github.com/carloslfu/slotstream) (MIT), and
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) (Apache-2.0).
See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The Ornith weights
are published by [ornith-ai](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit)
under their own terms.

## Results

All numbers are from an **Apple M3 Pro with 18 GB unified memory**, running
the 4-bit (~18 GB) qpack with `--cache-gb 2` unless noted. Output was
verified coherent, and byte-identical to the previous step where the
change allowed it. They are directional measurements from a single
machine, not a formal benchmark suite.

### Decode speed

| Stage | Decode tok/s | Change |
|---|---|---|
| First working port (serial expert-cache fill) | 5.7 – 5.8 | baseline |
| Concurrent expert-cache fill (bounded parallel `pread`) | 8.89 | +54% |
| Measurement fix + min-heap LFU eviction | 9.4 – 9.98 | |
| MoE kernel fusion (up to 38 dispatches per layer → 9) | 9.85 – 9.94 | +5 – 6% |
| CPU attention core vectorized (`cblas_sgemv` / vDSP) | 10.27 – 11.12 | +4 – 11%, larger at longer context |
| Base model, current defaults | **10.7 – 11.8** | **~2× the first port** |

Router-aware expert prefetch (on by default) guesses layer *i+1*'s experts
from layer *i*'s hidden state and has them loading while the GPU works. The
guess is right about 79% of the time at top-8 and 94% at top-16. The gain
grows with the size of the expert cache:

| `--cache-gb` | Cache hit rate (off → on) | Speedup |
|---|---|---|
| 2 | 55% → 71% | +4.7% |
| 4 | 77% → 85% | +11.9% |
| 6 | 88% → 93% | **+26.0%** |

(This sweep ran on a slower machine state, with `F_NOCACHE` on, so treat
the relative gains as meaningful but not the absolute tok/s.)

### Memory usage

Swiftlet streams routed experts from disk, so the full ~18 GB of weights
never has to be resident. Memory budget for a 32K-token context, from the
memory governor's own math:

| Component | Size |
|---|---|
| Dense / always-resident weights | 1.29 GiB |
| DeltaNet recurrent state | 62.8 MiB |
| KV cache @ 32K tokens (FP32; INT8 is now the default and smaller) | 1.25 GiB |
| Scratch reserve | 256 MiB |
| Safety margin | 8% of target |
| **Minimum viable total** | **~3.13 GiB** |

Any memory above that floor goes to the expert cache:

| Target memory | Expert cache | Resident experts (of 10,240) |
|---|---|---|
| 4 GiB | 0.82 GiB | 500 |
| 6 GiB | 2.66 GiB | 1,616 |
| 8 GiB | 4.50 GiB | 2,733 |

Measured peak process footprint during decode:

| `--cache-gb` | Peak footprint |
|---|---|
| 2 | 3.8 GB |
| 4 | 5.9 GB |
| 6 | 8.1 GB |
| 8 | 10.2 GB |

**Long context (~20–25K generated tokens).** INT8 KV-cache quantization,
now the default, gave **+14% decode throughput** (4.24 → 4.84 tok/s) and
cut **peak footprint by 20%** (7.06 GB → 5.67 GB). Quality stayed
near-lossless (K/V cosine similarity > 0.9999).

### Tried and rejected

These ideas were measured and did not help, so they were not shipped:
Metal untracked hazard mode, overlapping cache fill with GPU dispatch,
3-bit expert weights, and `F_NOCACHE` as a default. `F_NOCACHE` looked like
a +14% win at first, then regressed badly on a later machine state and was
reverted. Details for each are in
[`CONVERSION_PLAN.md`](CONVERSION_PLAN.md).

## Known limitations

- Text-only. There is no vision support, and MTP speculative decoding is
  not wired into the decode loop yet. The MTP draft head loads and runs;
  the draft → verify loop isn't built.
- Raising `--cache-gb` above 2 improves the hit rate a lot but improves
  wall-clock time much less on this 18 GB machine. Memory-compressor
  pressure is the partially confirmed cause.
- The abliterated "CRACK" GGUF variant noticeably degrades code generation
  compared with the official base checkpoint. This comes from that
  fine-tune, not from the port. Use the base model for code.

## Start here

- **[`synopsis.md`](synopsis.md)** covers what this project is, why the
  GGUF checkpoint needed converting before Swiftlet could load it, and a
  running log of what shipped.
- **[`CONVERSION_PLAN.md`](CONVERSION_PLAN.md)** has the detailed
  GGUF → MLX → qpack pipeline, plus full write-ups and benchmark numbers for
  each optimization.
- **[`handoff.md`](handoff.md)** tracks status from session to session, for
  whoever picks the work up next.
- **[`PORT.md`](PORT.md)** documents the standalone Swift port kit: what's
  in it, its memory and traffic calculations, and how to build it, test it,
  and apply the Swiftlet overlay.

## Layout

- **`Sources/`, `Tests/`**: the standalone Swift package
  (`OrnithSwiftletPort`). It contains the architecture profile, checkpoint
  compatibility checks, expert-cache policy, memory governor, prefill
  planner, and router reference code. It builds on Linux and macOS without
  model weights; see [`PORT.md`](PORT.md).
- **`swiftlet-overlay/`**: the integration overlay (source, patches, and an
  installer), meant to be copied into a Swiftlet checkout.
- **`design/`**: port map, validation gates, and benchmark plan.
- **`research/`**: background research notes behind the optimization work.
- **`scratch/`**: benchmark and sweep scripts with their logs, backing the
  numbers above. Not needed to build or run anything.

## Dependencies

This repo does not vendor [Swiftlet](https://github.com/leonickson1/Swiftlet).
Clone it as a sibling directory (`../Swiftlet`, which the `scratch/` scripts
assume), or point their `SWIFTLET`/`MODEL` environment variables at your own
build. Model weights (the Ornith GGUF/MLX checkpoints and the `.qpack` files
made from them) are never committed; see `synopsis.md` and `.gitignore` for
how to regenerate them locally.

## License

Copyright 2026 Hoa Ton-That. The original port code, overlay patches, and
documentation in this repo are licensed under Apache-2.0; see
[LICENSE](LICENSE). Swiftlet and the other projects credited above keep
their own copyrights and licenses.
