"""Does UNDO (ACTION7) actually reverse an action, per local game?

For every locally available game: reset, take simple actions until one changes the
board, then UNDO once and compare the board to the pre-probe board. Also notes
whether UNDO is in the engine's available_actions and whether a no-op action's
UNDO disturbs the board. Output: one line per game.

    .venv/bin/python scripts/test_undo_offline.py [game ...]
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine.enums import GameAction


def grid_of(frame):
    arr = np.array(frame.frame)
    while arr.ndim > 2:
        arr = arr[-1]
    return arr.astype(int).tolist()


def main():
    arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    envs = arcade.get_environments()
    ids = sorted(e.game_id.split("-")[0] for e in envs)
    if len(sys.argv) > 1:
        ids = [g for g in ids if g in sys.argv[1:]]
    print(f"{'game':<6} {'undo_avail':>10} {'probe':>6} {'changed':>7} {'undo_reverted':>13} {'noop_undo_ok':>12}  note")
    for gid in ids:
        env = arcade.make(gid)
        if env is None:
            print(f"{gid:<6} no env"); continue
        f = env.reset()
        avail = list(getattr(f, "available_actions", []) or [])
        undo_avail = 7 in [int(getattr(a, "value", a)) if not hasattr(a, "id") else int(a.id) for a in avail] if avail else None
        g0 = grid_of(f)
        probe = None; changed = False; reverted = None; note = ""
        noop_name = None
        for a in (1, 2, 3, 4, 5):
            if a not in [int(getattr(x, "value", x)) if not hasattr(x, "id") else int(x.id) for x in avail] and avail:
                continue
            f1 = env.step(GameAction[f"ACTION{a}"], None)
            if f1 is None:
                note = "env None"; break
            g1 = grid_of(f1)
            if g1 != g0:
                probe = a; changed = True
                f2 = env.step(GameAction.ACTION7, None)
                g2 = grid_of(f2) if f2 is not None else None
                reverted = (g2 == g0)
                if not reverted and g2 is not None:
                    diff = sum(1 for r in range(len(g0)) for c in range(len(g0[0])) if g0[r][c] != g2[r][c])
                    same_as_probe = (g2 == g1)
                    note = f"after UNDO differs from start in {diff} cells" + (" (= probe board, UNDO was a no-op)" if same_as_probe else "")
                break
            else:
                noop_name = noop_name or a
                g0 = g1  # unchanged anyway
        # does UNDO after a no-op action disturb the board?
        noop_undo_ok = None
        if noop_name is not None and changed:
            env2 = arcade.make(gid); fa = env2.reset(); ga = grid_of(fa)
            fb = env2.step(GameAction[f"ACTION{noop_name}"], None); gb = grid_of(fb)
            if gb == ga:
                fc = env2.step(GameAction.ACTION7, None); gc = grid_of(fc) if fc is not None else None
                noop_undo_ok = (gc == ga)
        print(f"{gid:<6} {str(undo_avail):>10} {str(probe):>6} {str(changed):>7} {str(reverted):>13} {str(noop_undo_ok):>12}  {note}")


if __name__ == "__main__":
    main()
