"""ONE serving change on the byte-identical Franzen M2 copy, by exact string substitution.

    python scripts/build_franzen_variant.py streams12   # MAXREQ/CUDAGRAPH_MAXBS 10 -> 12, ARC3_MAX_ACTIVE_STREAMS 10 -> 12
    python scripts/build_franzen_variant.py hicache48   # + --enable-hierarchical-cache --hicache-size 48 (sirikilohit's flags)

Every substitution must match exactly once or the build fails; all other cells stay byte-identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks-copy-franzen-m2"

VARIANTS = {
    "streams12": [
        (12, "    MAXREQ=10,\n", "    MAXREQ=12,\n"),
        (12, "    CUDAGRAPH_MAXBS=10,\n", "    CUDAGRAPH_MAXBS=12,\n"),
        (4, "'ARC3_MAX_ACTIVE_STREAMS': 10,", "'ARC3_MAX_ACTIVE_STREAMS': 12,"),
    ],
    "hicache48": [
        (12, '"--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging"]',
             '"--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging",\n'
             '    "--enable-hierarchical-cache", "--hicache-size", "48", "--hicache-write-policy", "write_through", "--hicache-io-backend", "kernel"]'),
    ],
}


def main() -> None:
    name = sys.argv[1]; subs = VARIANTS[name]
    nb = json.loads((BASE / "duck.ipynb").read_text()); cells = nb["cells"]
    for idx, old, new in subs:
        src = "".join(cells[idx]["source"])
        assert src.count(old) == 1, f"cell {idx}: expected exactly one match for {old[:40]!r}, got {src.count(old)}"
        src = src.replace(old, new)
        if cells[idx]["cell_type"] == "code" and not any(l.lstrip().startswith(("!", "%")) for l in src.splitlines()):
            compile(src, f"cell{idx}", "exec")   # cells with IPython magics cannot be compiled as plain Python
        cells[idx]["source"] = [l + "\n" for l in src.splitlines()]
    out = ROOT / f"notebooks-franzen-{name}"; out.mkdir(exist_ok=True)
    (out / "duck.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((BASE / "kernel-metadata.json").read_text())
    meta["id"] = f"tantan0327/arc3-franzen-{name}"; meta["title"] = f"ARC3 franzen {name}"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    base = json.loads((BASE / "duck.ipynb").read_text())["cells"]
    changed = [i for i in range(len(cells)) if "".join(cells[i]["source"]) != "".join(base[i]["source"])]
    print(f"built {out}: cells changed {changed} (expected {sorted(set(i for i,_,_ in subs))}), kernel {meta['id']}")


if __name__ == "__main__":
    main()
