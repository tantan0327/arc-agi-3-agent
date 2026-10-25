"""Stack ONE patch cell on the byte-identical Franzen M2 copy, after the customisation hook.

The copy's cells: 0-1 markdown, 2 his patch (applied at setup), 10 sys.path, 12 serving,
14 benchmark unpickle, 16 customisation hook (bm.* overrides), 20 `await bm.run(...)`.
A code cell inserted at index 17 runs after the bundle is importable and before the
run, which is where a monkey-patch of `inference.agent.tool_agent` belongs. Kernel
metadata is copied from the Franzen copy with only id/title changed.

    python scripts/build_franzen_arm.py replay --patch scripts/patches/solver_replay.py [--env K=V ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks-copy-franzen-m2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--patch", required=True); ap.add_argument("--env", action="append", default=[]); ap.add_argument("--patient", action="store_true", help="45-min server wait + 40-min first-request grace (as in dprime-patient)")
    args = ap.parse_args()
    nb = json.loads((BASE / "duck.ipynb").read_text())
    cells = nb["cells"]
    assert len(cells) == 23, len(cells)
    assert "bm.solver.max_runtime_s_per_game" in "".join(cells[16]["source"]), "cell 16 is not the customisation hook"
    assert "await bm.run(" in "".join(cells[20]["source"]), "cell 20 is not the run cell"
    src = Path(args.patch).read_text(); compile(src, args.patch, "exec")
    env_lines = "".join("import os; os.environ[%r] = %r\n" % tuple(kv.split("=", 1)) for kv in args.env)
    header = "# arm %s: ONE patch on the Franzen M2 copy (%s); everything else byte-identical.\n" % (args.name, Path(args.patch).name)
    if args.patient:
        c4 = "".join(cells[4]["source"])
        assert "SERVER_STARTUP_TIMEOUT = 12 * 60" in c4 and "os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900'" in c4, "cell 4 startup lines not found"
        c4 = c4.replace("SERVER_STARTUP_TIMEOUT = 12 * 60", "SERVER_STARTUP_TIMEOUT = 45 * 60  # patient: slow Kaggle storage; the wait loop exits as soon as /health answers")
        c4 = c4.replace("os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900'", "os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '2400'  # patient: 40-min first-request grace")
        cells[4]["source"] = [l + "\n" for l in c4.splitlines()]
    cells.insert(17, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [l + "\n" for l in (header + env_lines + src).splitlines()]})
    out = ROOT / f"notebooks-franzen-{args.name}"; out.mkdir(exist_ok=True)
    (out / "duck.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((BASE / "kernel-metadata.json").read_text())
    meta["id"] = f"tantan0327/arc3-franzen-{args.name}"; meta["title"] = f"ARC3 franzen {args.name}"; meta["code_file"] = "duck.ipynb"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    compile("".join(cells[17]["source"]), "cell17", "exec")
    print(f"built {out}: {len(cells)} cells, patch cell 17 = {len(src)} chars, kernel {meta['id']}")


if __name__ == "__main__":
    main()
