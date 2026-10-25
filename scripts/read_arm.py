"""Read demo runs of the Franzen chassis against the base band.

    python3 scripts/read_arm.py scratchpad/arc3-franzen-census_out [more dirs]

Band = the three identical-config base runs (copy-franzen-m2, rep, rep2). Prints per run:
levels, actions, model turns (from the requests files), and per game levels vs the band range;
then the adoption read: levels above band max? games up/down? actions <= 1.2x base mean?
"""
from __future__ import annotations
import glob, json, os, sys

BAND_DIRS = ["scratchpad/arc3-copy-franzen-m2_out", "scratchpad/arc3-franzen-rep_out", "scratchpad/arc3-franzen-rep2_out"]


def read(d):
    p = os.path.join(d, "benchmark.json")
    if not os.path.exists(p):
        return None
    runs = json.load(open(p))["game_runs"]
    per = {}
    for r in runs:
        g = r["game_id"][:4]
        per[g] = {"levels": int(r.get("levels_completed") or 0), "actions": len(r.get("history") or []),
                  "score": float(r.get("final_score") or 0.0), "state": r.get("state")}
    turns = {}
    for f in glob.glob(os.path.join(d, "*_requests.jsonl")):
        g = os.path.basename(f)[:4]
        try:
            turns[g] = sum(1 for _ in open(f, errors="replace"))
        except Exception:
            turns[g] = None
    for g in per:
        per[g]["turns"] = turns.get(g)
    return per


def main():
    band = [b for b in (read(d) for d in BAND_DIRS) if b]
    games = sorted(set().union(*[set(b) for b in band])) if band else []
    rng = {g: (min(b.get(g, {}).get("levels", 0) for b in band), max(b.get(g, {}).get("levels", 0) for b in band)) for g in games}
    band_levels = [sum(v["levels"] for v in b.values()) for b in band]
    band_actions = [sum(v["actions"] for v in b.values()) for b in band]
    print("band: levels %s (max %d)  actions %s (mean %.0f)" % (band_levels, max(band_levels), band_actions, sum(band_actions) / len(band_actions)))
    for d in sys.argv[1:]:
        per = read(d)
        if not per:
            print(d, "no benchmark.json"); continue
        L = sum(v["levels"] for v in per.values()); A = sum(v["actions"] for v in per.values())
        T = sum(v["turns"] or 0 for v in per.values()); S = sum(v["score"] for v in per.values()) / max(1, len(per))
        up = [g for g in per if per[g]["levels"] > rng.get(g, (0, 0))[1]]
        down = [g for g in per if per[g]["levels"] < rng.get(g, (0, 0))[0]]
        print("%s: levels %d  actions %d  turns %d  mean score %.2f" % (os.path.basename(d), L, A, T, S))
        print("  per game: " + "  ".join("%s %d(%d-%d)" % (g, per[g]["levels"], *rng.get(g, (0, 0))) for g in sorted(per)))
        print("  above band max: %s | games up: %s | games below range: %s | actions <= 1.2x base: %s" % (
            L > max(band_levels), up or "-", down or "-", A <= 1.2 * sum(band_actions) / len(band_actions)))


if __name__ == "__main__":
    main()
