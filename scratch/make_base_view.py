#!/usr/bin/env python3
"""Build a disk-cheap 'view' of hf-checkpoint/ with block 40's tensors
stripped out, for a normal (non-MTP) mlx_lm.convert run. Every shard file
except the one holding layer-40 tensors is symlinked (zero extra disk);
that one shard is rewritten without the 20 layer-40 keys. Non-safetensors
files (config.json etc.) are symlinked too.
"""
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("hf-checkpoint")
DST = Path("hf-checkpoint-base")
DST.mkdir(exist_ok=True)

index = json.loads((SRC / "model.safetensors.index.json").read_text())
wm = index["weight_map"]
mtp_keys = {k for k in wm if k.startswith("model.layers.40.")}
mtp_shards = sorted(set(wm[k] for k in mtp_keys))
print(f"{len(mtp_keys)} MTP keys in shard(s): {mtp_shards}")

for f in SRC.iterdir():
    if f.name == "model.safetensors.index.json":
        continue
    dst = DST / f.name
    if dst.exists() or dst.is_symlink():
        continue
    if f.name in mtp_shards:
        continue  # handled specially below
    dst.symlink_to(f.resolve())

for shard_name in mtp_shards:
    src_path = SRC / shard_name
    tensors = {}
    with safe_open(str(src_path), framework="pt") as sf:
        for k in sf.keys():
            if k in mtp_keys:
                continue
            tensors[k] = sf.get_tensor(k)
    dst_path = DST / shard_name
    save_file(tensors, str(dst_path), metadata={"format": "pt"})
    print(f"rewrote {shard_name}: {len(tensors)} tensors (dropped {len(mtp_keys)})")

new_wm = {k: v for k, v in wm.items() if k not in mtp_keys}
new_index = {
    "metadata": {"total_size": sum((DST / f).stat().st_size for f in set(new_wm.values()))},
    "weight_map": new_wm,
}
(DST / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2))
print(f"wrote index: {len(new_wm)} tensors, {len(set(new_wm.values()))} shards")
print(f"done: {DST.resolve()}")
