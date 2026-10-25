"""Stack ONE patch on the Thuitanium B81 reproduction, everything else byte-identical.

The B81 notebook (`notebooks-thui-b81-asis/duck.ipynb`) has the floor's cell
layout: cell 9 makes the anim solver importable and imports `tool_agent`;
cells 10-11 load the benchmark; cell 15 runs it. The patch goes in as a new
code cell at index 10 -- after the import, before the run -- exactly where
candidate B's note backfill sat on the floor. Kernel metadata is copied from
the B81 dir with only id/title/code_file changed, so the attached datasets,
model, image and machine are identical.

    python scripts/build_stack_on_b81.py cand-c --patch scripts/patches/note_from_reasoning.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks-thui-b81-asis"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--patch", required=True)
    args = ap.parse_args()
    nb = json.loads((BASE / "duck.ipynb").read_text())
    assert len(nb["cells"]) == 18, len(nb["cells"])
    assert "THUI_ANIMFAST_GRAFT" in "".join(nb["cells"][9]["source"]), "cell 9 is not the B81 graft cell"
    src = Path(args.patch).read_text()
    compile(src, args.patch, "exec")
    header = f"# candidate {args.name}: ONE patch on the B81 chassis ({Path(args.patch).name}); everything else byte-identical.\n"
    nb["cells"].insert(10, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                            "source": [l + "\n" for l in (header + src).splitlines()]})
    out = ROOT / f"notebooks-{args.name}"
    out.mkdir(exist_ok=True)
    (out / "duck.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((BASE / "kernel-metadata.json").read_text())
    meta["id"] = f"tantan0327/arc3-{args.name}"; meta["title"] = f"ARC3 {args.name}"; meta["code_file"] = "duck.ipynb"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    compile("".join(nb["cells"][10]["source"]), "cell10", "exec")
    print(f"built {out}: {len(nb['cells'])} cells, patch cell 10 = {len(src)} chars, kernel {meta['id']}")


if __name__ == "__main__":
    main()
