"""Replay the recorded human solutions offline and capture the winning moments.

The premise being tested, before anything is built on it: a level's goal is the
same across that game's levels, so a description extracted when the human
finishes level 1 should also describe how they finish levels 2..N. If that does
not hold, "learn the goal from the first completion" is not a design.

The replays are ground truth -- Tufa's CI asserts they reach WIN scoring exactly
100.0 -- so any desync here is our environment differing from theirs, not the
human being wrong, and is reported as such rather than papered over.
"""
import json, glob, os, sys, logging
logging.disable(logging.INFO)
import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine.enums import GameAction

# The replays ship only in the 2.3 MB winning bundle, not the 450 KB share one:
#   kaggle datasets download -d jeroencottaar/taaf-kaggle-source --unzip -p <dir>
REPLAYS = os.environ.get(
    "HUMAN_REPLAYS",
    "vendor/taaf-kaggle-source/src/re-arc-3/re_arc/dsl/official_human_replays")
CAPS = os.environ.get("REPLAY_CAPS", "scratchpad/caps")


def grid_of(frame):
    """The playfield as plain nested ints.

    `frame.frame` is (layers, 64, 64) -- a list of numpy rows, so the obvious
    `isinstance(g, list)` unwrapping walks straight past it. Go through numpy
    and take the last layer, which is the one drawn on top.
    """
    g = getattr(frame, "frame", None)
    if g is None:
        return None
    arr = np.array(g)
    while arr.ndim > 2:
        arr = arr[-1]
    return arr.astype(int).tolist()


def replay(arcade, short_id, levels):
    env = arcade.make(short_id)
    if env is None:
        return None, "no env"
    f = env.reset()
    seen_levels = f.levels_completed
    out = [[-1, None, grid_of(f)]]   # seed: opening board of level 0
    total = 0
    for li, lvl in enumerate(levels):
        for a, p in lvl:
            before = grid_of(f)
            data = {"x": p["x"], "y": p["y"]} if a == 6 and p else None
            f = env.step(GameAction[f"ACTION{a}"], data)
            total += 1
            if f is None:
                return out, f"env returned None after {total} actions"
            if f.levels_completed > seen_levels:
                out.append([seen_levels, before, grid_of(f)])
                seen_levels = f.levels_completed
    return out, f"levels_completed={seen_levels}/{len(levels)} after {total} actions"


def main():
    arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    files = sorted(glob.glob(f"{REPLAYS}/*.json"))
    if len(sys.argv) > 1:
        files = [f for f in files if os.path.basename(f)[:-5] in sys.argv[1:]]
    print(f"{'game':<7} {'completions captured':>21}  status")
    ok = 0
    for f in files:
        g = os.path.basename(f)[:-5]
        levels = [[(a, p) for a, p in lvl] for lvl in json.load(open(f))["levels"]]
        try:
            caps, status = replay(arcade, g, levels)
        except Exception as exc:
            print(f"{g:<7} {'-':>21}  EXC {type(exc).__name__}: {str(exc)[:60]}")
            continue
        n = (len(caps) - 1) if caps else 0   # minus the seeded opening row
        if n == len(levels):
            ok += 1
        print(f"{g:<7} {f'{n}/{len(levels)}':>21}  {status}")
        json.dump(caps or [], open(f"{CAPS}/{g}.json", "w"))
    print(f"\nfully in sync: {ok}/{len(files)}")


if __name__ == "__main__":
    os.makedirs(CAPS, exist_ok=True)
    main()
