#!/usr/bin/env python3
"""Speculative-decoding scoping question: if we batch-verify K draft
positions in one pass, how many DISTINCT routed experts does layer L need
to fetch across those K positions, vs. the 8 it needs for just one?
Classical speculative decoding's "K tokens cost ~1" argument assumes the
same weights get read regardless of batch width -- true for this model's
dense parts, not obviously true for the routed experts (each position can
pick a different top-8 of 256). This answers that from real router traces
(SWIFTLET_ROUTER_TRACE=<path>, QwenMetalModel.RouterTraceRecorder) instead
of guessing.

Usage: python3 router_overlap_analysis.py <trace.csv> [<trace2.csv> ...]
Trace format (no header): position,layer,expert;expert;...;expert (top-K,
sorted ascending).
"""
import sys
from collections import defaultdict

def load(path):
    # layer -> list of (position, frozenset(experts)), sorted by position
    by_layer = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pos_s, layer_s, picks_s = line.split(",", 2)
            picks = frozenset(int(x) for x in picks_s.split(";") if x)
            by_layer[int(layer_s)].append((int(pos_s), picks))
    for layer in by_layer:
        by_layer[layer].sort(key=lambda t: t[0])
    return by_layer

def analyze(by_layer, ks=(1, 2, 4, 8)):
    topk = None
    for layer, seq in by_layer.items():
        if seq:
            topk = len(seq[0][1])
            break
    results = {}  # k -> (avg_distinct, layers analyzed, windows analyzed)
    for k in ks:
        distinct_sum = 0.0
        windows = 0
        for layer, seq in by_layer.items():
            n = len(seq)
            for i in range(n - k + 1):
                union = set()
                for j in range(i, i + k):
                    union |= seq[j][1]
                distinct_sum += len(union)
                windows += 1
        avg_distinct = distinct_sum / windows if windows else float("nan")
        results[k] = (avg_distinct, windows)
    return topk, results

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ks = (1, 2, 4, 8)
    print(f"{'file':<40} {'K':>3} {'avg distinct experts':>22} {'vs 8K (disjoint)':>18} {'vs 8 (identical)':>18}")
    for path in sys.argv[1:]:
        by_layer = load(path)
        topk, results = analyze(by_layer, ks)
        for k in ks:
            avg_distinct, windows = results[k]
            disjoint = topk * k
            frac_of_disjoint = avg_distinct / disjoint
            frac_over_identical = avg_distinct / topk
            print(f"{path:<40} {k:>3} {avg_distinct:>22.2f} {frac_of_disjoint:>17.1%} {frac_over_identical:>17.2f}x")
        print()

if __name__ == "__main__":
    main()
