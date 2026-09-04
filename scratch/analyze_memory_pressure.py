#!/usr/bin/env python3
"""Analyze a memory_pressure_sweep.sh output directory.

For each cachegb_<N>_run<R> in the sweep dir, correlates:
  - the existing "decode Metal S3a" wait-minus-exec gap (the thing
    CONVERSION_PLAN.md's "--cache-gb gap revisited" section left as an
    untested memory-pressure question)
  - per-process footprint.json (phys_footprint / phys_footprint_peak)
  - system-wide vm_stat.txt (pageins/pageouts/swapins/swapouts/compressor
    activity, summed over the run; free-page gauge, first vs. last)

Usage: ./analyze_memory_pressure.py <sweep-dir>
"""
import json
import re
import sys
from pathlib import Path

WAIT_RE = re.compile(r"^decode Metal S3a:.*wait=([\d.]+)s/(\d+).*gpu=([\d.]+)s/(\d+)", re.MULTILINE)
CACHE_RE = re.compile(r"expert cache: (\d+) slots \(([\d.]+) GB\), (\d+) hits / (\d+) misses \(([\d.]+)% hit rate\)")
TOKS_RE = re.compile(r"decode: (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)")

VMSTAT_COLS = [
    "free", "active", "specul", "inactive", "throttle", "wired", "prgable",
    "faults", "copy", "0fill", "reactive", "purged", "file-backed",
    "anonymous", "cmprssed", "cmprssor", "dcomprs", "comprs", "pageins",
    "pageout", "swapins", "swapouts",
]
COUNTER_COLS = {"faults", "copy", "0fill", "reactive", "purged", "cmprssor",
                 "dcomprs", "comprs", "pageins", "pageout", "swapins", "swapouts"}


def parse_num(tok: str) -> int:
    tok = tok.rstrip(".")
    if tok.endswith("K"):
        return int(float(tok[:-1]) * 1000)
    return int(tok)


def parse_stderr(path: Path):
    text = path.read_text()
    wait = WAIT_RE.search(text)
    cache = CACHE_RE.search(text)
    toks = TOKS_RE.search(text)
    out = {}
    if wait:
        wait_s, wait_n, gpu_s, gpu_n = wait.groups()
        out["wait_s"] = float(wait_s)
        out["gpu_s"] = float(gpu_s)
        out["gap_s"] = float(wait_s) - float(gpu_s)
    if cache:
        out["slots"] = int(cache.group(1))
        out["cache_gib"] = float(cache.group(2))
        out["hit_rate_pct"] = float(cache.group(5))
    if toks:
        out["decode_tok_s"] = float(toks.group(3))
    return out


def parse_vmstat(path: Path):
    lines = [l for l in path.read_text().splitlines() if l.strip() and not l.startswith("Mach")]
    rows = []
    prev_was_header = False
    for l in lines:
        parts = l.split()
        if len(parts) != len(VMSTAT_COLS):
            continue
        try:
            values = [parse_num(p) for p in parts]
        except ValueError:
            # The column-header line ("free active specul ...") re-prints
            # periodically in vm_stat's interval mode; it has the same
            # token count as a data row but fails int parsing. The row
            # immediately following each header reprint is a fresh
            # cumulative-since-report-start value, not a per-second delta
            # (confirmed empirically: one such row was ~2.67e9 faults,
            # matching the true first row's cumulative value, and threw
            # off pageins_sum by 1000x before this fix) -- mark it so it
            # gets excluded from delta sums same as row 0.
            prev_was_header = True
            continue
        rows.append((values, prev_was_header))
        prev_was_header = False
    if len(rows) < 2:
        return {}
    # First row overall, and the row right after every header reprint, are
    # cumulative snapshots, not per-second deltas -- exclude from sums.
    deltas = [v for i, (v, after_header) in enumerate(rows) if i > 0 and not after_header]
    if not deltas:
        return {}
    idx = {c: i for i, c in enumerate(VMSTAT_COLS)}
    out = {}
    for c in COUNTER_COLS:
        out[f"{c}_sum"] = sum(r[idx[c]] for r in deltas)
    out["free_first"] = deltas[0][idx["free"]]
    out["free_last"] = deltas[-1][idx["free"]]
    out["cmprssed_last"] = deltas[-1][idx["cmprssed"]]
    out["samples"] = len(deltas)
    return out


def parse_footprint(path: Path):
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return {}
    samples = d.get("samples", [])
    if not samples:
        return {}
    peaks, finals = [], []
    for s in samples:
        procs = s.get("processes", [])
        if not procs:
            continue
        aux = procs[0].get("auxiliary", {})
        if "phys_footprint_peak" in aux:
            peaks.append(aux["phys_footprint_peak"])
        if "phys_footprint" in aux:
            finals.append(aux["phys_footprint"])
    out = {}
    if peaks:
        out["phys_footprint_peak_mb"] = max(peaks) / 1e6
    if finals:
        out["phys_footprint_last_mb"] = finals[-1] / 1e6
    out["footprint_samples"] = len(samples)
    return out


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sweep_dir = Path(sys.argv[1])
    stderrs = sorted(sweep_dir.glob("cachegb_*.stderr"))
    if not stderrs:
        print(f"no cachegb_*.stderr files found in {sweep_dir}")
        sys.exit(1)

    rows = []
    for se in stderrs:
        name = se.stem
        row = {"run": name}
        row.update(parse_stderr(se))
        vm = se.with_name(name + ".vmstat.txt")
        if vm.exists():
            row.update(parse_vmstat(vm))
        fp = se.with_name(name + ".footprint.json")
        if fp.exists():
            row.update(parse_footprint(fp))
        rows.append(row)

    cols = ["run", "slots", "hit_rate_pct", "decode_tok_s", "wait_s", "gpu_s",
            "gap_s", "phys_footprint_peak_mb", "free_first", "free_last",
            "pageins_sum", "pageout_sum", "swapins_sum", "swapouts_sum",
            "cmprssor_sum", "dcomprs_sum"]
    widths = {c: max(len(c), 8) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, "-"))))

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "-")).ljust(widths[c]) for c in cols))

    print()
    print("Interpretation guide:")
    print("  gap_s = wait_s - gpu_s, the previously-documented "
          "GPU-wait-minus-real-exec gap (2.76s->4.83s in the original "
          "CRACK-build sweep at cache slot counts 1213->4854).")
    print("  If phys_footprint_peak_mb / free_last-shrinking / "
          "pageins_sum+swapouts_sum grow in the same shape as gap_s across "
          "these rows, that supports the memory-pressure hypothesis.")
    print("  If those signals are flat while gap_s still grows, memory "
          "pressure (at least as footprint/vm_stat can see it) is ruled "
          "out too, same as the hazard-tracking hypothesis was.")


if __name__ == "__main__":
    main()
