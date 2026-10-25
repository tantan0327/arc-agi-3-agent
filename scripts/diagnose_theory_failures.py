"""Classify the transitions the avatar theory fails to predict, per game, to see which
primitives the theory space is missing (push, rotation, non-rigid change, multi-mover...)."""
from __future__ import annotations
import glob, json, os, sys, collections, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ti", ROOT / "scripts/test_theory_induction.py"); ti = importlib.util.module_from_spec(spec); spec.loader.exec_module(ti)
import numpy as np, logging; logging.disable(logging.CRITICAL)
import arc_agi
from arc_agi import OperationMode
from arcengine.enums import GameAction


def classify(before, after, pred_off):
    changed = np.argwhere(before != after)
    n = len(changed)
    mv = ti.movers(before, after)
    if n == 0:
        return "no_change(pred move)" if pred_off is not None else "no_change"
    if mv:
        offs = set(mv.values())
        if pred_off is None:
            return "moved(pred blocked)"
        if pred_off in offs and len(mv) > 1:
            return "multi_mover(avatar ok)"
        if pred_off not in offs:
            return "moved_other_offset"
        return "moved_as_predicted?"
    # no rigid translation found: characterise the non-rigid change
    cols_b = set(before[tuple(changed.T)].tolist()); cols_a = set(after[tuple(changed.T)].tolist())
    if n <= 4:
        return "tiny_nonrigid(<=4 cells)"
    if len(cols_b | cols_a) <= 2 and n >= 100:
        return "large_recolour/reveal"
    return "nonrigid(%d cells)" % min(n, 999)


def main():
    arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    games = sys.argv[1:] or ["tu93", "sc25", "g50t", "tr87", "bp35", "ar25"]
    for g in games:
        levels = [[(a, p) for a, p in lvl] for lvl in json.load(open(f"{ti.REPLAYS}/{g}.json"))["levels"]]
        env = arcade.make(g); f = env.reset(); before = ti.grid_of(f)
        th = ti.RichTheory() if os.environ.get('TI_CLASS') == 'rich' else ti.AvatarTheory(); fails = collections.Counter(); total = 0; counted = 0
        for lvl in levels:
            for a, p in lvl:
                data = {"x": p["x"], "y": p["y"]} if a == 6 and p else None
                f = env.step(GameAction[f"ACTION{a}"], data); after = ti.grid_of(f)
                if a in ti.DIRS:
                    total += 1
                    pred = th.predict(before, a, p)
                    if pred is not None:
                        counted += 1
                        c, ok = (ti.evaluate_rich if os.environ.get('TI_CLASS') == 'rich' else ti.evaluate_avatar)(before, after, a, pred)
                        if not ok:
                            fails[classify(before, after, pred[1])] += 1
                    else:
                        fails["no_opinion:" + ("no_change" if not (before != after).any() else ("moved" if ti.movers(before, after) else "nonrigid"))] += 1
                th.observe(before, after, a, p); before = after
        print(f"== {g}: direction transitions {total}, with opinion {counted}; failure classes:")
        for k, v in fails.most_common(8):
            print(f"     {v:4d}  {k}")


if __name__ == "__main__":
    main()
