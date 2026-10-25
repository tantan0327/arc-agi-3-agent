"""Theory-based planning test (EMPA step 3): fit the core-knowledge theory on the human's
level 1, infer the goal colour from that completion, then let a BFS planner in the theory
play levels 2..N in the REAL engine — plan, act, compare to prediction, re-plan on surprise.

Reports per game: levels the planner completed, actions used vs the human baseline, re-plans.
No model, no tokens. A level the planner fails is replayed from the human solution so the
next level can still be tested.

    .venv/bin/python scripts/test_theory_planner.py [game ...]
"""
from __future__ import annotations
import glob, json, os, sys, collections, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ti", ROOT / "scripts/test_theory_induction.py"); ti = importlib.util.module_from_spec(spec); spec.loader.exec_module(ti)
import numpy as np, logging; logging.disable(logging.CRITICAL)
import arc_agi
from arc_agi import OperationMode
from arcengine.enums import GameAction

MAX_PLAN_DEPTH = 80
MAX_REPLANS = 6
MAX_ACTIONS_PER_LEVEL = 150


def front_colours(grid, cells, u, k):
    H, W = grid.shape
    sweep = set()
    cur = set(cells)
    for i in range(1, k + 1):
        nxt = {(r + u[0], c + u[1]) for r, c in cur}
        sweep |= (nxt - set(cells))
        cur = nxt
    return {int(grid[r, c]) for r, c in sweep if 0 <= r < H and 0 <= c < W}, all(0 <= r < H and 0 <= c < W for r, c in cur)


def plan(theory, grid, goal_colour):
    """BFS over avatar placements in the static board under the fitted theory.
    Returns a list of keys, or None."""
    keys = list(ti.DIRS) if any(theory.abs_dir.values()) else []
    if not keys or theory.avatar is None:
        return None
    seen_steps = [theory.step[a].most_common(1)[0][0] for a in ti.DIRS if theory.step[a]]
    default_k = int(collections.Counter(seen_steps).most_common(1)[0][0]) if seen_steps else 1
    start = frozenset(theory.avatar)
    seen = {start}; queue = collections.deque([(start, [])])
    while queue:
        cells, path = queue.popleft()
        if len(path) >= MAX_PLAN_DEPTH:
            continue
        for a in keys:
            u = theory.predicted_unit(a)
            if u is None:
                continue
            k = int(theory.step[a].most_common(1)[0][0]) if theory.step[a] else default_k
            cols, inside = front_colours(grid, cells, u, k)
            if not inside:
                continue
            if goal_colour in cols:
                return path + [a]
            if any(theory.block[a][d] > theory.passed[a][d] for d in cols):
                continue
            nxt = frozenset((r + u[0] * k, c + u[1] * k) for r, c in cells)
            if nxt in seen:
                continue
            seen.add(nxt); queue.append((nxt, path + [a]))
    return None


def goal_from_completion(theory, before, a):
    if a not in ti.DIRS or theory.avatar is None or not theory.abs_dir[a]:
        return None
    u = theory.abs_dir[a].most_common(1)[0][0]
    kf = theory.step[a].most_common(1)[0][0] if theory.step[a] else 1
    cols, _ = front_colours(before, theory.avatar, u, int(kf))
    cols = sorted(cols)
    return cols


def run_game(arcade, gid, levels, goal_override=None):
    env = arcade.make(gid)
    if env is None:
        return None
    f = env.reset(); before = ti.grid_of(f)
    th = ti.RichTheory()
    # level 1: the human plays, the theory watches; the goal colours come from its completing action
    goal_cols = None
    for ai, (a, p) in enumerate(levels[0]):
        if ai == len(levels[0]) - 1:
            goal_cols = goal_from_completion(th, before, a)
        f = env.step(GameAction[f"ACTION{a}"], {"x": p["x"], "y": p["y"]} if a == 6 and p else None)
        after = ti.grid_of(f); th.observe(before, after, a, p); before = after
    if goal_override is not None:
        goal_cols = [goal_override]
    out = {"game": gid, "goal": goal_cols, "levels": []}
    if not goal_cols:
        return out
    counts = collections.Counter(before.flatten().tolist())
    bg = counts.most_common(1)[0][0]
    cands = [c for c in goal_cols if c != bg] or goal_cols
    goal_colour = min(cands, key=lambda c: counts.get(c, 0))
    for li in range(1, len(levels)):
        human_n = len(levels[li]); done = f.levels_completed
        th.relocate(before)
        actions = 0; replans = 0; completed = False; surprises = 0
        while actions < MAX_ACTIONS_PER_LEVEL and replans <= MAX_REPLANS:
            th.relocate(before)
            path = plan(th, before, goal_colour)
            if not path:
                break
            replans += 1
            diverged = False
            for a in path:
                pred = th.predict(before, a, None)
                f = env.step(GameAction[f"ACTION{a}"], None); actions += 1
                if f is None:
                    break
                after = ti.grid_of(f); th.observe(before, after, a, None)
                moved = ti.movers_any(before, after)
                before = after
                if f.levels_completed > done:
                    completed = True; break
                # surprise: predicted a move and nothing moved, or vice versa -> re-plan
                if pred is not None and ((pred[1] is None) != (not moved)):
                    surprises += 1; diverged = True; break
                if actions >= MAX_ACTIONS_PER_LEVEL:
                    break
            if completed or f is None:
                break
            if not diverged:
                break   # the plan ran to its end without the level completing: the goal theory is wrong here
        out["levels"].append({"level": li + 1, "completed": completed, "actions": actions, "human": human_n, "replans": replans, "surprises": surprises})
        if not completed:
            # resync: replay the human's actions for this level (the planner's actions may have moved things;
            # the human's sequence from a different state may not complete it either -> stop the game)
            f = env.step(GameAction.RESET, None) if hasattr(GameAction, "RESET") else f
            before = ti.grid_of(f) if f is not None else before
            ok = False
            for a, p in levels[li]:
                f = env.step(GameAction[f"ACTION{a}"], {"x": p["x"], "y": p["y"]} if a == 6 and p else None)
                if f is None:
                    break
                after = ti.grid_of(f); th.observe(before, after, a, p); before = after
                if f.levels_completed > done:
                    ok = True; break
            if not ok:
                out["stopped_at"] = li + 1
                break
        else:
            done = f.levels_completed
    return out


def main():
    arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    files = sorted(glob.glob(f"{ti.REPLAYS}/*.json"))
    want = set(sys.argv[1:])
    print(f"{'game':<6} {'goal':>6} | levels completed by the planner / tested | actions planner:human | re-plans | detail")
    tot_done = tot_test = 0
    for fp in files:
        g = os.path.basename(fp)[:-5]
        if want and g not in want:
            continue
        levels = [[(a, p) for a, p in lvl] for lvl in json.load(open(fp))["levels"]]
        if not any(a in ti.DIRS for a, _ in levels[0]):
            continue   # click-only game: no avatar to plan with
        try:
            out = run_game(arcade, g, levels)
        except Exception as exc:
            print(f"{g:<6} EXC {type(exc).__name__}: {str(exc)[:80]}"); continue
        if not out or not out["levels"]:
            print(f"{g:<6} {str(out.get('goal') if out else None):>6} | no plannable levels"); continue
        done = sum(1 for l in out["levels"] if l["completed"]); tested = len(out["levels"])
        pa = sum(l["actions"] for l in out["levels"] if l["completed"]); ph = sum(l["human"] for l in out["levels"] if l["completed"])
        detail = " ".join(f"L{l['level']}:{'OK' if l['completed'] else 'x'}({l['actions']}/{l['human']})" for l in out["levels"])
        print(f"{g:<6} {str(out['goal']):>6} | {done}/{tested} | {pa}:{ph} | {sum(l['replans'] for l in out['levels'])} | {detail}")
        tot_done += done; tot_test += tested
    print(f"\nALL: planner completed {tot_done} of {tot_test} tested levels (levels 2+ of direction games), no model, no tokens")


if __name__ == "__main__":
    main()
