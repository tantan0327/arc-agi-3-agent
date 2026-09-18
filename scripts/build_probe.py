"""Build a THROUGHPUT PROBE kernel from the floor vehicle.

A nine-hour Save & Run costs a quarter of the weekly GPU quota and answers
"how many levels" with an instrument whose noise (sd ~0.9) swamps most
changes. But every lever we have left works through one quantity — how many
model turns a game gets inside its clock — and that is measurable in thirty
minutes:

    requests / minute, prompt tokens / request, time to first token

So a probe runs the same 25 public games on the same bundle, model and
serving stack, with `max_runtime_s_per_game` cut to 1,800 s. Boot (~10 min)
plus play (~30 min) costs about 0.75 h of quota, so six variants cost less
than one confirmation run. Levels are reported too, but the throughput
numbers are the reading.

    python scripts/build_probe.py <name> [--env K=V ...] [--patch FILE] [--seconds 1800]

Writes notebooks-probe-<name>/ with kernel id tantan0327/arc3-probe-<name>.
`--patch` inlines a Python file into a cell that runs AFTER the bundle is
importable and BEFORE the benchmark, so the harness can be monkey-patched
without forking the bundle dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks-flashnext-asis" / "duck.ipynb"
META = ROOT / "notebooks-flashnext-asis" / "kernel-metadata.json"


def build(name: str, env: dict, patch: str | None, seconds: int) -> Path:
    nb = json.loads(BASE.read_text())

    # 1. environment overrides. They MUST be applied after the solver's setup
    # commands (cell 9), which persist their own LOCAL_ANALYZER_* / MULTIMODAL_*
    # values into the environment and would otherwise overwrite ours -- that is
    # why the first context-window arm measured nothing. The agent module reads
    # these at import time, and the import happens later, so a cell inserted
    # right after the setup is both late enough and early enough.
    pre = []
    if env:
        pre = [f"# PROBE {name}: environment overrides, applied AFTER the solver setup",
               "import json, os"]
        pre += [f'os.environ[{k!r}] = {v!r}' for k, v in env.items()]
        pre += [f'print("PROBE_ENV {name} " + json.dumps({env!r}, sort_keys=True), flush=True)', ""]

    # 2. the shortened clock, in the settings cell
    cell13 = "".join(nb["cells"][13]["source"])
    old = "bm.solver.max_runtime_s_per_game = 7920.0"
    assert old in cell13
    cell13 = cell13.replace(old, f"bm.solver.max_runtime_s_per_game = {float(seconds)}  # PROBE: throughput, not depth", 1)
    nb["cells"][13]["source"] = cell13.splitlines(keepends=True)

    # 3. one inserted cell carrying the env overrides and any monkey-patch,
    # placed after the bundle is importable and its setup has run (cell 9).
    src_lines = list(pre)
    if patch:
        src_lines += Path(patch).read_text().splitlines()
    if src_lines:
        nb["cells"].insert(10, {"cell_type": "code", "metadata": {}, "execution_count": None,
                                "outputs": [], "source": [l + "\n" for l in src_lines]})

    out = ROOT / f"notebooks-probe-{name}"
    out.mkdir(exist_ok=True)
    (out / "duck.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads(META.read_text())
    meta["id"] = f"tantan0327/arc3-probe-{name}"
    meta["title"] = f"arc3 probe {name}"
    meta["is_private"] = True
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE, repeatable")
    ap.add_argument("--patch", help="python file inlined as a post-import cell")
    ap.add_argument("--seconds", type=int, default=1800)
    args = ap.parse_args()
    env = {}
    for item in args.env:
        k, _, v = item.partition("=")
        env[k] = v
    out = build(args.name, env, args.patch, args.seconds)
    print(f"built {out} (env {env}, patch {args.patch}, {args.seconds}s/game)")


if __name__ == "__main__":
    main()
