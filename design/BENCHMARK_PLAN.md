# Benchmark and acceptance plan

## Gate 0: package and overlay tests

- `swift test` passes in this standalone package.
- Install the overlay in a clean Swiftlet checkout and run its complete test
  suite on macOS.
- No existing Swiftlet test may be disabled or have its tolerance widened.

## Gate 1: checkpoint and qpack integrity

Repack the official MLX INT4 checkpoint. Record:

- dense `model.safetensors` size;
- expert section names, dtypes, shapes, offsets, payload, and stride;
- 40 layer files, 256 expert records per layer;
- qpack manifest verification result.

Expected default geometry:

```text
payload = stride = 1,769,472 bytes per routed expert
routed pool = 18,119,393,280 bytes
```

This baseline intentionally hard-fails if the official MLX INT4/group-64
payload differs. A different quantization must add an explicit profile with its
own section geometry and numerical parity tests; native K=8 is never inferred
away.

## Gate 2: reference parity

Use greedy decoding and identical tokenized prompts in mlx-lm and Swiftlet.
Capture token IDs, not only rendered strings.

Recommended corpus:

1. one-token prompt;
2. prose completion;
3. source-code completion;
4. long system prompt plus short question;
5. multi-turn chat;
6. prompt that emits a reasoning block;
7. prompt that terminates on each declared EOS token.

Acceptance:

- exact greedy token IDs for the agreed comparison length, or a documented
  first-logit tolerance and identical top-1 when CPU/GPU floating-point order
  prevents bit identity;
- exact top-8 expert IDs per layer on a router trace;
- selected router weights sum to one when `norm_topk_prob` is omitted.

## Gate 3: cache semantics

Run the same greedy corpus with cache budgets spanning the practical range:

```text
minimum 16 slots (the current upstream `ExpertCache` floor)
0.25 GiB
0.5 GiB
1 GiB
2 GiB
4 GiB
fully resident when hardware permits
```

Acceptance:

- cache size changes performance and memory only;
- generated token IDs do not change;
- every requested top-8 batch is simultaneously resident before dispatch;
- no in-flight expert slot is reused early;
- no short reads are ignored.

## Gate 4: performance attribution

For each hardware target and cache setting, collect:

- prefill tokens/s and time to first token;
- decode tokens/s after at least 64 generated tokens;
- peak process footprint and Metal allocated size;
- expert slots, hits, misses, and read bytes/token;
- `pread` wall time and queue depth;
- command buffers, blocking waits, dispatches, and GPU execution time/token;
- thermal state for sustained runs.

Interpretation:

- high miss rate with little wait time: do not buy RAM or add I/O complexity;
- visible read stalls and idle GPU: prototype bounded parallel reads;
- many tiny dispatches with busy GPU: fuse/persist the MoE kernel;
- prefill near decode speed: prioritize real batched prefill;
- low MTP acceptance: stop; do not hide quality loss behind aggregate tok/s.

## Gate 5: optimization experiments

Run one controlled change at a time against the same prompt/token corpus.

1. Cache policy trace simulation: current LFU+recency, LRU, CLOCK, and
   frequency decay.
2. Bounded read widths: 1, 2, 4, 8; record SSD throughput and CPU overhead.
3. Shared-expert/read overlap.
4. Fused gate+up and then persistent top-8 MoE.
5. Prefill chunk sizes selected by the memory planner.
6. MTP proposal lengths 1, 2, and 4 with acceptance distribution.

Reject an optimization that changes native K, generated tokens, or memory
accounting unless it is exposed as a separate quality-altering mode.

## Example cache sweep

The included `scripts/cache-sweep.sh` invokes Swiftlet repeatedly and stores
stdout/stderr per cache size. Edit the model and prompt paths first.
