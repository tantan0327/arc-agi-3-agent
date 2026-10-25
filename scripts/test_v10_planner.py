"""v10 mechanical layer, offline: how many human actions does planning over an
abstraction save?

Per-level score is min((baseline/actions)^2, 1.15) -- CAPPED -- and on the
levels the agent already completes it sits at ~0.69x human actions, i.e. at
the cap, so action savings on reached levels do not raise the score directly.
They matter through the clock (every game hits 7,920 s): actions not spent
wandering are time for deeper levels, priced by the measured concave clock
curve (4x time -> 2.51x score). Astra saves 51.7% of human actions. This test measures what OUR mechanical
layer -- transition memory over abstract states + BFS -- saves, using the
GP-B hand-written abstractions as stand-ins for authored ones.

Protocol per game, per level segment of the human trajectory:
  1. observe the human's transitions in the level (state_k, action_k -> state_k+1)
     as abstract edges -- exactly what the harness would have recorded;
  2. BFS from the level's first abstract state to its last (the state the
     human's final action left, i.e. the completing move) over recorded edges;
  3. compare the planned path length to the human's action count.

The planner can only reuse observed edges, so savings come from NOT repeating
exploration -- backtracking, detours, wall bumps -- the "known-state" part of
Astra's gain. Generalisation (delta rules on progression games) is measured
separately on s5i5.

    python scripts/test_v10_planner.py
"""
from __future__ import annotations

import gzip
import pickle
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gpb_reference_abstractions import ABSTRACTIONS  # noqa: E402


def level_key(gid, state):
    # every GP-B reference puts the level identity first
    return state[0]


def bfs(edges, start, goal, max_nodes=200000):
    if start == goal:
        return 0
    seen = {start}; q = deque([(start, 0)])
    while q and len(seen) < max_nodes:
        s, d = q.popleft()
        for a, nxt in edges.get(s, ()):
            if nxt == goal:
                return d + 1
            if nxt not in seen:
                seen.add(nxt); q.append((nxt, d + 1))
    return None


def main():
    games = pickle.load(gzip.open(ROOT / "scratchpad" / "replay_cache.pkl.gz"))
    total_h = total_p = 0
    for gid, decisions in games:
        fn = ABSTRACTIONS.get(gid[:4])
        if fn is None:
            continue
        keys = [fn(b.tolist()) for b, _, _ in decisions]
        akeys = [(n, c) if c else n for _, n, c in decisions]
        # segment by level identity
        segs = []; start = 0
        for i in range(1, len(keys)):
            if level_key(gid, keys[i]) != level_key(gid, keys[i - 1]):
                segs.append((start, i)); start = i
        segs.append((start, len(keys) - 1))
        rows = []
        for a, b in segs:
            if b - a < 3:
                continue
            edges = defaultdict(list)
            for i in range(a, b):
                edges[keys[i]].append((akeys[i], keys[i + 1]))
            human = b - a
            # goal = the state reached by the human's last action of the segment
            planned = bfs(edges, keys[a], keys[b])
            rows.append((human, planned))
        h = sum(r[0] for r in rows); p = sum(r[1] for r in rows if r[1] is not None)
        solved = sum(1 for r in rows if r[1] is not None)
        total_h += h; total_p += p
        print(f"{gid[:4]}: {len(rows)} level segments, planner re-solved {solved}; "
              f"human actions {h} -> planned {p}  (saves {1 - p / h:.0%})")
        for human, planned in rows[:6]:
            print(f"    level: human {human:3d}  planned {planned}")
    print(f"\nALL (reference abstractions): human {total_h} -> planned {total_p}  "
          f"= {1 - total_p / total_h:.0%} fewer actions from replanning over observed edges alone")
    print("Score is efficiency squared: a saving of s on reached levels multiplies "
          "their score by 1/(1-s)^2.")


if __name__ == "__main__":
    main()
