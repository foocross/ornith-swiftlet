#!/bin/sh
set -eu

# Long-context sweep for the KV-cache INT8 quantization feature
# (KVQuant.swift, SWIFTLET_KV_QUANT=int8), FP32 baseline vs INT8, at the
# 32K-token context length synopsis.md's own memory math already targets.
# This is the piece the approved plan flagged as not pre-existing anywhere
# in the repo (BENCHMARK_PLAN.md's Gate 3 is a cache-BUDGET sweep, not a
# context-LENGTH sweep) -- authored here rather than found.
#
# Runs FP32 then INT8 sequentially (not concurrently -- resource
# contention on one machine would confound the throughput comparison),
# each ~32768 tokens at ~11 tok/s on this machine (~50 min/run, ~100 min
# total), sampling per-process footprint every 5s throughout to get a real
# measurement of the KV memory savings at the context length where it
# actually matters (at ~200 tokens, per earlier same-day testing, the KV
# cache is too small a fraction of total footprint to see the difference
# at all).
#
# Usage: cd scratch && ./kv_quant_long_context_sweep.sh
# Output: scratch/kv_quant_long_sweep-<timestamp>/{fp32,int8}.{stdout,stderr,footprint.json}

SWIFTLET=${SWIFTLET:-../Swiftlet/.build/release/swiftlet}
MODEL=${MODEL:-ornith-1.5-35b-base.qpack}
MAXNEW=${MAXNEW:-32768}
PROMPT="Write an extremely long, detailed, comprehensive essay about the entire history of the Netherlands from prehistoric times to the present day. Cover every major period in depth: prehistoric settlement, the Roman frontier, the Frankish era, the medieval County of Holland and the Low Countries, the Burgundian and Habsburg periods, the Eighty Years' War and the Dutch Republic, the Dutch Golden Age (trade, art, science, the VOC and WIC, global exploration and colonization), the 18th century decline, the Batavian Republic and Napoleonic period, the Kingdom of the Netherlands from 1815 onward, industrialization, the World Wars and occupation, decolonization, post-war reconstruction and the welfare state, European integration, and developments into the 21st century. Do not summarize -- go into extensive, specific detail with dates, names, and events for each era, writing many thousands of words."
OUT=${OUT_DIR:-kv_quant_long_sweep-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"

if [ ! -x "$SWIFTLET" ]; then
    echo "error: $SWIFTLET not found or not executable" >&2
    exit 1
fi

run_one() {
    label=$1
    quant=$2  # "" for FP32 default, "int8" to set SWIFTLET_KV_QUANT
    echo "=== $label starting $(date) (max-new $MAXNEW) ===" >&2
    if [ -n "$quant" ]; then
        SWIFTLET_KV_QUANT="$quant" "$SWIFTLET" generate "$MODEL" \
            --gpu --chat --cache-gb 2 --max-new "$MAXNEW" --prompt "$PROMPT" \
            >"$OUT/${label}.stdout" 2>"$OUT/${label}.stderr" &
    else
        "$SWIFTLET" generate "$MODEL" \
            --gpu --chat --cache-gb 2 --max-new "$MAXNEW" --prompt "$PROMPT" \
            >"$OUT/${label}.stdout" 2>"$OUT/${label}.stderr" &
    fi
    pid=$!
    footprint -p "$pid" --sample 5 -j "$OUT/${label}.footprint.json" >/dev/null 2>&1 &
    fppid=$!

    wait "$pid" || echo "  warning: $label exited non-zero" >&2
    sleep 0.3
    kill -INT "$fppid" 2>/dev/null || true
    wait "$fppid" 2>/dev/null || true

    echo "=== $label done $(date) ===" >&2
    grep -E "decode:|decode Metal S3a|expert cache:" "$OUT/${label}.stderr" >&2 || true
}

run_one fp32 ""
run_one int8 "int8"

echo "results in: $OUT" >&2
