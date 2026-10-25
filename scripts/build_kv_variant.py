"""B81 with ONE serving change: the vLLM KV cache size (cell 3), nothing else.

Why: the +1.3 the B81 chassis brought over our floor came from a serving
profile (MTP off, KV 5 -> 7 GiB, 28 seqs) that halves the request queue.
Thuitanium measured KV 7/10/12 GiB with MTP off as all booting (Running 10/14/17
at 1,800 s) but never ran 12 GiB at full length. This builds that arm.

    python scripts/build_kv_variant.py 12   -> notebooks-b81-kv12/, kernel tantan0327/arc3-b81-kv12
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks-thui-b81-asis"


def main() -> None:
    gib = int(sys.argv[1])
    nb = json.loads((BASE / "duck.ipynb").read_text())
    src = "".join(nb["cells"][3]["source"])
    subs = [("PUBLIC25_VLLM_PROFILE_NAME = 'kv7-bf16-mtp0-c28-cg32'", f"PUBLIC25_VLLM_PROFILE_NAME = 'kv{gib}-bf16-mtp0-c28-cg32'"),
            ('"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "7516192768"', f'"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "{gib * 1024 ** 3}"'),
            ("== str(7 * 1024 ** 3)", f"== str({gib} * 1024 ** 3)"),
            ('print("THUI_A5_PROFILE ok mtp=0 kv=7GiB seqs=28", flush=True)', f'print("THUI_A5_PROFILE ok mtp=0 kv={gib}GiB seqs=28  # kv variant, one change vs B81", flush=True)')]
    for old, new in subs:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    compile(src, "cell3", "exec")
    nb["cells"][3]["source"] = [l + "\n" for l in src.splitlines()]
    out = ROOT / f"notebooks-b81-kv{gib}"
    out.mkdir(exist_ok=True)
    (out / "duck.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((BASE / "kernel-metadata.json").read_text())
    meta["id"] = f"tantan0327/arc3-b81-kv{gib}"; meta["title"] = f"ARC3 b81 kv{gib}"; meta["code_file"] = "duck.ipynb"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    other = [i for i in range(18) if i != 3 and "".join(nb["cells"][i]["source"]) != "".join(json.loads((BASE / "duck.ipynb").read_text())["cells"][i]["source"])]
    print(f"built {out}: cell 3 changed, other cells changed: {other}, kernel {meta['id']}")


if __name__ == "__main__":
    main()
