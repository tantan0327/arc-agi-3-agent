"""Build a trace-collection variant of the Franzen copy: which games to EXCLUDE from the
demo and the per-game wall cap in minutes. Everything else byte-identical.

    python scripts/build_hardtrace.py hardtrace2a --exclude ar25,ft09,... --minutes 60
"""
import argparse, json, glob, os
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--exclude", default=""); ap.add_argument("--minutes", type=int, default=25)
ap.add_argument("--patch", default=None, help="python file to insert as a cell before the run cell")
ap.add_argument("--env", action="append", default=[], help="K=V lines prepended to the patch cell")
a = ap.parse_args()
src = glob.glob("notebooks-copy-franzen-m2/*.ipynb")[0]; nb = json.load(open(src))
s = "".join(nb["cells"][16]["source"])
old = [l for l in s.splitlines() if l.strip().startswith("demo_excluded_games =")][0]
excl = [g.strip() for g in a.exclude.split(",") if g.strip()]
new = "demo_excluded_games = %r  # %s: trace collection" % (excl, a.name)
s2 = s.replace(old, new)
old_cap = "bm.solver.max_runtime_s_per_game = 25*60 #532*60 * bm.solver.concurrency // 110"
assert old_cap in s2, "cap line changed"
s2 = s2.replace(old_cap, "bm.solver.max_runtime_s_per_game = %d*60  # %s: longer per-game wall for deeper traces" % (a.minutes, a.name))
assert s2 != s
nb["cells"][16]["source"] = s2.splitlines(keepends=True)
assert len(nb["cells"]) == 23
if a.patch:
    assert "await bm.run(" in "".join(nb["cells"][20]["source"]), "cell 20 is not the run cell"
    env_lines = "".join("import os; os.environ[%r] = %r\n" % tuple(kv.split("=", 1)) for kv in a.env)
    src = env_lines + open(a.patch).read()
    compile(src, "patch", "exec")
    nb["cells"].insert(17, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)})
d = "notebooks-franzen-%s" % a.name; os.makedirs(d, exist_ok=True)
json.dump(nb, open(os.path.join(d, "duck.ipynb"), "w"), indent=1)
meta = json.load(open("notebooks-copy-franzen-m2/kernel-metadata.json"))
meta["id"] = "tantan0327/arc3-franzen-%s" % a.name; meta["title"] = "ARC3 franzen %s" % a.name; meta["code_file"] = "duck.ipynb"
json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
print("built", d, "| games run:", 25 - len(excl), "| cap", a.minutes, "min | excluded:", excl)
