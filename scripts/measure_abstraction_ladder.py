"""Does the right abstraction turn these games from perception into search?

measure_search_vs_perception.py found 98.3% of human decisions occur at a
masked-BOARD state never seen before — no revisitable structure, so
harness-level graph search is dead AT THAT REPRESENTATION. But GPT-6 Astra's
traces (ARC Prize's analysis) show it abstracts further: its self-invented DSL
tracks object-level state ("Turn 5: P=(24,20), empty, facing west"; "L8: hub
q2. Lengths: 14=1"). At that level, states that are pixel-novel may be
familiar, and transition structure may become reusable.

This script walks the same 6,512 human decisions and measures state novelty
at each rung of an abstraction ladder:

  A0  masked board digest (the original measurement, 98.3% novel)
  A1  object set: sorted (colour, component_count) — what objects exist
  A2  object layout: sorted (colour, centroid//4) — where objects are, coarse
  A3  object census + avatar: (colour counts) + the moving object's position

If novelty collapses down the ladder while the games stay distinguishable,
the deficit is not search and not memory — it is the abstraction step
(perception -> symbolic state), which is exactly what Core Knowledge object
priors supply humans and what Astra's DSL re-derives per game. That sizes the
Grand Prize architecture: mechanical search over LEARNED-ABSTRACTION states,
with the model needed only to author the abstraction.

    python scripts/measure_abstraction_ladder.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLAYS = ROOT / "vendor/taaf-kaggle-source/src/re-arc-3/re_arc/dsl/official_human_replays"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/taaf-kaggle-source/src/re-arc-3"))

from duck_delta.worldmodel_kit import ChromeMask  # noqa: E402


def components(board, colour):
    """4-connected component count and centroids for one colour, no deps."""
    import numpy as np
    grid = np.asarray(board)
    mask = grid == colour
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols \
                                and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                ys = sum(y for y, _ in cells) / len(cells)
                xs = sum(x for _, x in cells) / len(cells)
                comps.append((len(cells), round(ys), round(xs)))
    return comps


def digests(board, cm):
    """One digest per abstraction rung for this board."""
    import numpy as np
    grid = np.asarray(board)
    colours = sorted(int(c) for c in np.unique(grid))
    per_colour = {c: components(board, c) for c in colours}

    a0 = cm.digest(np.asarray(board).tolist())
    a1 = tuple(sorted((c, len(per_colour[c])) for c in colours))
    a2 = tuple(sorted((c, cy // 4, cx // 4)
                      for c in colours for _, cy, cx in per_colour[c]))
    # A3: colour census plus the position of the rarest small object (the
    # avatar proxy -- the human-click study found small rare blobs are what
    # humans track).
    rare = None
    for c in colours:
        for size, cy, cx in per_colour[c]:
            if size <= 6:
                cand = (len(per_colour[c]), size, c, cy, cx)
                if rare is None or cand < rare:
                    rare = cand
    census = tuple(sorted((c, sum(s for s, _, _ in per_colour[c]))
                          for c in colours))
    a3 = (census, (rare[3], rare[4]) if rare else None)
    return {"A0 masked board": a0, "A1 object set": a1,
            "A2 object layout": a2, "A3 census+avatar": a3}


def main() -> None:
    from re_arc import EnvSampler
    from arcengine import GameAction
    import numpy as np
    by_value = {m.value: m for m in GameAction}

    rungs = ["A0 masked board", "A1 object set", "A2 object layout",
             "A3 census+avatar"]
    novel = {r: 0 for r in rungs}
    revisit_edge = {r: 0 for r in rungs}   # state seen AND this action tried
    total = 0
    for p in sorted(REPLAYS.glob("*.json")):
        data = json.loads(p.read_text())
        gid = data["game_id"]
        try:
            env = EnvSampler(include=[gid]).make(gid)
        except Exception:
            continue
        obs = env.reset()
        cm = ChromeMask()
        seen: dict[str, dict] = {r: {} for r in rungs}
        prev = None
        for actions in data["levels"]:
            for value, adata in actions:
                ga = by_value.get(int(value))
                if ga is None:
                    break
                board = np.array(obs.frame[-1]).tolist()
                if prev is not None:
                    cm.learn(prev, board)
                ds = digests(board, cm)
                total += 1
                for r in rungs:
                    key = repr(ds[r])
                    tried = seen[r].get(key)
                    if tried is None:
                        novel[r] += 1
                        seen[r][key] = {ga.name}
                    else:
                        if ga.name in tried:
                            revisit_edge[r] += 1
                        tried.add(ga.name)
                prev = board
                obs, _r, _d, _i = env.step(ga, adata or None)

    print(f"human decisions: {total}")
    print(f"{'rung':<20} {'novel state':>12} {'known state+action':>20}")
    for r in rungs:
        print(f"{r:<20} {novel[r]/total:>11.1%} {revisit_edge[r]/total:>19.1%}")
    print("\nReading: A0 is the old result. If novelty falls sharply by A2/A3,")
    print("a compact object-level state space exists in which transition")
    print("structure is reusable -- the wall is the abstraction step, not")
    print("search. That is what Astra's DSL supplies at $19-26k per game.")


if __name__ == "__main__":
    main()
