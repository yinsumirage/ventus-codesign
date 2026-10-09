"""Software-only control v2: frozen H0, full prefill menu incl. block order."""
import json, sys
sys.path.insert(0, ".")
from codesign.ventus import qwen_search as qs
designs = [[{}, sw] for sw in qs.phase_menu("prefill", qs.TILES)]
json.dump(designs, open("results/abc-study/designs/swonly2-prefill.json", "w"), indent=1)
designs = [[{}, sw] for sw in qs.phase_menu("decode", qs.TILES)]
json.dump(designs, open("results/abc-study/designs/swonly2-decode.json", "w"), indent=1)
print("ok")
