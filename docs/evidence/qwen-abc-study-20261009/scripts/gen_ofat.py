"""One-factor-at-a-time exploration samples around three centres (2/3/4 SMs)."""
import json, sys
sys.path.insert(0, ".")
from codesign.ventus.qwen_search import RANGES
from codesign.ventus.qwen_study import full_hardware
base = json.load(open("examples/baseline-hardware-v1.json"))
lean = dict(lds_bytes=16384, lds_banks=8, lds_ports=1, sgpr_slots=256)
C2 = dict(base, **lean, l1_mshrs=16, l2_mshrs=64, lsu_entries=16, lsu_per_warp=8)
C3 = json.load(open(sys.argv[1]))  # classmate's selected 3-SM design
C4 = dict(base, **lean, sms=4, collectors=4, tensor_m=2, tensor_n=8, tensor_k=2,
          l1_sets=64, l1_ways=1, l1_mshrs=16, l1_subentries=1, l2_sets=32, l2_ways=4,
          lsu_entries=16, lsu_per_warp=2)
SW = {"share_packing": True, "gqa_batch": True, "fuse_rows1": True}
skip = {"lds_bytes", "lds_banks", "lds_ports"}
designs, seen = [], set()
def add(hw):
    try:
        h = full_hardware({k: hw[k] for k in RANGES})
    except ValueError:
        return
    key = json.dumps(h, sort_keys=True)
    if key not in seen:
        seen.add(key); designs.append([{k: h[k] for k in RANGES}, SW])
for c in (C2, C3, C4):
    add(c)
    for k, vals in RANGES.items():
        if k in skip: continue
        for v in vals:
            if v != c[k]:
                add(dict(c, **{k: v}))
json.dump(designs, open(sys.argv[2], "w"), indent=0)
print(len(designs))
