"""How much of solving is search (a harness can do it) vs perception (only an
LLM can)? — the sizing test for v9 (harness-driven world model).

v8 asked the 27B to write predict() and it wouldn't. v9 flips the driver: the
HARNESS builds the empirical transition graph from observed (masked state,
action) -> state edges, and when the agent stands on a known state with a
known path to an unexplored frontier, it PLANS and executes with no LLM call
-- reserving the model for genuine novelty (unseen states, goal inference,
click targets). Rudakov placed 3rd training-free with exactly this shape.

This script sizes the opportunity on the human replays (the only 100%-optimal
trajectories we hold). Walking each game's human trajectory and building the
masked transition graph incrementally, every human action is classified:

- REDISCOVERABLE: from the current masked state, this action's base was
  already tried before from an equivalent state -- a harness that remembered
  would not need the model to choose it again.
- NOVEL: a state or action the graph has not seen -- genuinely needs the
  model.

The rediscoverable fraction is an upper bound on the decisions v9's harness
could take without the LLM. High = search carries most of the load and v9 has
real headroom; low = perception dominates and v9 cannot help much.

    python scripts/measure_search_vs_perception.py
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


def main() -> None:
    from re_arc import EnvSampler
    from arcengine import GameAction
    import numpy as np
    by_value = {m.value: m for m in GameAction}

    total = redisc = novel_state = novel_action = 0
    per_game = []
    for p in sorted(REPLAYS.glob("*.json")):
        data = json.loads(p.read_text())
        gid = data["game_id"]
        try:
            env = EnvSampler(include=[gid]).make(gid)
        except Exception:
            continue
        obs = env.reset()
        cm = ChromeMask()
        edges: dict[str, set[str]] = {}     # masked state -> action bases tried
        g_total = g_redisc = 0
        prev = None
        for actions in data["levels"]:
            for value, adata in actions:
                ga = by_value.get(int(value))
                if ga is None:
                    break
                board = np.array(obs.frame[-1]).tolist()
                if prev is not None:
                    cm.learn(prev, board)
                s = cm.digest(board)
                base = ga.name
                tried = edges.get(s)
                total += 1
                g_total += 1
                if tried is None:
                    novel_state += 1
                elif base in tried:
                    redisc += 1
                    g_redisc += 1
                else:
                    novel_action += 1
                edges.setdefault(s, set()).add(base)
                prev = board
                obs, _r, _d, _i = env.step(ga, adata or None)
        if g_total:
            per_game.append((gid[:4], g_redisc / g_total, g_total))

    print(f"human decisions: {total}")
    print(f"  rediscoverable (known state, action tried before): "
          f"{redisc} = {redisc/total:.1%}")
    print(f"  novel state (never seen this masked board):        "
          f"{novel_state} = {novel_state/total:.1%}")
    print(f"  novel action (known state, new action):            "
          f"{novel_action} = {novel_action/total:.1%}")
    print("\nper game (rediscoverable share, n):")
    for gid, frac, n in sorted(per_game, key=lambda r: -r[1]):
        print(f"  {gid:<6} {frac:5.1%}  ({n})")
    print("\nReading: the rediscoverable share is the ceiling on decisions v9's")
    print("harness could take without the model. High -> search has headroom;")
    print("low -> perception dominates and v9's driver-flip cannot help much.")


if __name__ == "__main__":
    main()
