"""What does approaching the goal LOOK like, behaviourally? (GP1)

The sharpest agent failure we have measured is that it stops like a coin
flip: 5.5x chance at starting a game and exactly chance at finishing one.
Humans finish everything. This script asks what the *behavioural signature*
of human progress is, using only action streams — no board understanding
required — so the answer can become a signal a harness can compute online.

Within each level, actions are split into thirds (early/mid/late) and three
statistics are computed per third:

- action-type entropy (bits): diverse action mixing = exploring; a narrow
  distribution = executing a plan;
- click dispersion: the std of MOUSE coordinates (row+col pooled), spatial
  convergence toward a target region;
- immediate-repeat rate: the share of actions equal to their predecessor
  (plan execution often repeats a direction; so does mindless wandering —
  read jointly with entropy).

Cohorts: every human level (all completed, by construction of the data);
agent levels that were completed; and the agent's stalled tail (the actions
spent on the level it never finished). If human thirds show falling entropy
and shrinking dispersion while the stalled tail stays flat, "progress" has a
measurable signature the agent's own stream lacks — and its absence is
detectable online, which no current harness exploits.

    python scripts/measure_progress_signal.py <agent_run_dir>
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLAYS = ROOT / "vendor/taaf-kaggle-source/src/re-arc-3/re_arc/dsl/official_human_replays"


def entropy(names: list[str]) -> float:
    if not names:
        return float("nan")
    c = Counter(names)
    n = len(names)
    return -sum((k / n) * math.log2(k / n) for k in c.values())


def dispersion(coords: list[tuple[int, int]]) -> float:
    if len(coords) < 3:
        return float("nan")
    ys = [c[0] for c in coords]
    xs = [c[1] for c in coords]
    my, mx = sum(ys) / len(ys), sum(xs) / len(xs)
    return math.sqrt(sum((y - my) ** 2 + (x - mx) ** 2 for y, x in zip(ys, xs))
                     / len(coords))


def repeat_rate(names: list[str]) -> float:
    if len(names) < 2:
        return float("nan")
    return sum(a == b for a, b in zip(names, names[1:])) / (len(names) - 1)


def thirds(seq):
    n = len(seq)
    if n < 6:
        return None
    a, b = n // 3, 2 * n // 3
    return seq[:a], seq[a:b], seq[b:]


def stats_for(segments):
    """segments: list of (names, coords) per level/tail."""
    out = []
    for third_idx in range(3):
        es, ds, rs = [], [], []
        for names, coords in segments:
            t = thirds(list(zip(names, coords)))
            if t is None:
                continue
            part = t[third_idx]
            pn = [p[0] for p in part]
            pc = [p[1] for p in part if p[1] is not None]
            e, d, r = entropy(pn), dispersion(pc), repeat_rate(pn)
            if not math.isnan(e):
                es.append(e)
            if not math.isnan(d):
                ds.append(d)
            if not math.isnan(r):
                rs.append(r)
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
        out.append((mean(es), mean(ds), mean(rs), len(es)))
    return out


def human_segments():
    segs = []
    for p in sorted(REPLAYS.glob("*.json")):
        data = json.loads(p.read_text())
        for actions in data["levels"]:
            names, coords = [], []
            for value, adata in actions:
                names.append(f"A{value}")
                coords.append((int(adata["y"]), int(adata["x"]))
                              if adata and adata.get("x") is not None else None)
            segs.append((names, coords))
    return segs


def agent_segments(run_dir: Path):
    bm = json.loads((run_dir / "benchmark.json").read_text())
    completed, stalled = [], []
    for g in bm["game_runs"]:
        hist = g.get("history") or []
        names, coords = [], []
        for h in hist:
            a = h.get("action") or {}
            names.append(str(a.get("id")))
            d = a.get("data") or {}
            coords.append((int(d["y"]), int(d["x"]))
                          if d.get("x") is not None else None)
        apl = g.get("actions_per_level") or []
        done = int(g.get("levels_completed") or 0)
        off = 0
        for k in range(done):
            n = int(apl[k] or 0)
            completed.append((names[off:off + n], coords[off:off + n]))
            off += n
        if off < len(names):
            stalled.append((names[off:], coords[off:]))
    return completed, stalled


def show(label, table):
    print(f"\n{label}")
    print(f"  {'third':<8}{'entropy(bits)':>14}{'click disp':>12}{'repeat':>9}{'n':>6}")
    for name, (e, d, r, n) in zip(("early", "mid", "late"), table):
        print(f"  {name:<8}{e:>14.3f}{d:>12.2f}{r:>9.2%}{n:>6}")


def main() -> None:
    run_dir = Path(sys.argv[1])
    hs = human_segments()
    comp, stall = agent_segments(run_dir)
    print(f"human levels: {len(hs)} | agent completed: {len(comp)} | "
          f"agent stalled tails: {len(stall)}")
    show("HUMAN (all levels, all completed)", stats_for(hs))
    show("AGENT, completed levels", stats_for(comp))
    show("AGENT, stalled tails (the level it never finished)", stats_for(stall))


if __name__ == "__main__":
    main()
