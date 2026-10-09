"""Software-only control: frozen H0 hardware, every phase software choice."""
import json, sys
sys.path.insert(0, ".")
from codesign.ventus import qwen_search as qs
for phase in ("prefill", "decode"):
    designs = [[{}, qs.software_for(phase, t)] for t in qs.TILES[phase]]
    json.dump(designs, open(f"results/abc-study/designs/swonly-{phase}.json", "w"), indent=1)
    print(phase, len(designs))
