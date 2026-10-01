"""End-to-end test of the goto tool in the REAL sandbox against the REAL engine.

For each game with a human replay: play the first k human actions of level 1
on the engine (that is the history the sandbox will see), pick a target the
human's body reaches m moves later, then run `result = goto(R, C)` through
`run_sandboxed_python` -- the actual restricted subprocess the agent's python
tool uses -- with an action handler that steps the engine exactly as the
harness does. Reports, per game: whether the rules were learned, whether the
target was reached, and actions taken against the human's m.

    PYTHONPATH=vendor/taaf-kaggle-source/src/ARC3-Inference python3 scripts/test_goto_tool.py [--games m0r0,re86] [--k 12] [--m 8]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor/taaf-kaggle-source/src/re-arc-3"))
sys.path.insert(0, str(ROOT / "vendor/taaf-kaggle-source/src/ARC3-Inference"))

from inference.agent.python_tool_sandbox import run_sandboxed_python  # noqa: E402
from duck_delta import goto_tool as GT  # noqa: E402

REPLAYS = ROOT / "vendor/taaf-kaggle-source/src/re-arc-3/re_arc/dsl/official_human_replays"
PRELUDE = (ROOT / "duck_delta" / "goto_tool.py").read_text()
NAME = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 6: "MOUSE"}
ENG = {"UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3", "RIGHT": "ACTION4", "SPACE": "ACTION5", "MOUSE": "ACTION6"}


def frame_payload(grid, step, level):
    return {"grid_b64": base64.b64encode(bytes(int(v) for row in grid for v in row)).decode("ascii"),
            "shape": [len(grid), len(grid[0])], "step": step, "level": level}


def run_game(gid, data, k, m, log):
    from re_arc import EnvSampler
    from arcengine import GameAction
    by_value = {a.value: a for a in GameAction}; by_name = {a.name: a for a in GameAction}
    env = EnvSampler(include=[gid]).make(gid); obs = env.reset()
    grid = lambda o: [list(map(int, r)) for r in o.frame[-1]]
    level = lambda o: int(getattr(o, "levels_completed", 0) or 0) + 1
    history = []                       # (action_text, grid, level)
    lvl1 = data["levels"][0]
    # adapt the window to short levels: learn from the first half, target inside the rest
    if len(lvl1) < 8:
        return {"gid": gid[:4], "skip": f"level 1 has only {len(lvl1)} actions"}
    k = min(k, max(4, len(lvl1) // 2)); m = min(m, len(lvl1) - k - 1)
    # 1. the first k human actions become the recorded history
    for value, adata in lvl1[:k]:
        obs, *_ = env.step(by_value[int(value)], adata or None)
        name = NAME.get(int(value), str(value))
        text = f"MOUSE(row={adata['y']}, col={adata['x']})" if name == "MOUSE" and adata else name
        history.append((text, grid(obs), level(obs)))
    # 2. the target: where the human's body is m moves later (computed with the tool's own learner)
    class E:  # minimal history entry / frame views for the offline learner
        def __init__(s, a, g, l): s.action = a; s.frame = type("F", (), {"_grid": g, "level": l})()
    hist_views = [E(a, g, l) for a, g, l in history]
    model = GT._learn(GT._transitions(hist_views))
    if model is None:
        return {"gid": gid[:4], "learned": False}
    probe = EnvSampler(include=[gid]).make(gid); o2 = probe.reset()
    for value, adata in lvl1[:k + m]:
        o2, *_ = probe.step(by_value[int(value)], adata or None)
    if level(o2) != level(obs):
        return {"gid": gid[:4], "learned": True, "skip": "human completed the level inside the window"}
    tgt = GT._locate(grid(o2), model)
    if tgt is None:
        return {"gid": gid[:4], "learned": True, "skip": "no body at the target frame"}
    target = (tgt[0], tgt[1])
    start = GT._locate(grid(obs), model)
    # offline feasibility with the learned rules, before touching the engine
    diag = {"floor": model["floor"], "passable": sorted(model["passable"]), "blocking": sorted(model["blocking"]),
            "n_bodies": model["n_bodies"], "shapes": len(model["shapes"])}
    if start is not None:
        st = GT._static(grid(obs), model); s0 = (start[0], start[1])
        ps = GT._plan(st, s0, start[2], model, lambda p, key: p == target)
        po = GT._plan(st, s0, start[2], model, lambda p, key: p == target, optimistic=True)
        diag["plan_strict"] = None if ps is None else len(ps); diag["plan_optimistic"] = None if po is None else len(po)
    # 3. the sandbox, driven exactly like the harness
    step_counter = {"n": k}

    def state_payload(last=None):
        return {"current_frame": frame_payload(grid(obs), step_counter["n"], level(obs)),
                "history": [{"action": a, "frame": frame_payload(g, i + 1, l)} for i, (a, g, l) in enumerate(history)],
                "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"],
                "last_action_result": last or {}}

    def handler(actions):
        nonlocal obs
        executed = 0; changed = False; lvl_done = False; over = False
        before_level = level(obs)
        for a in actions:
            name = str(a.get("action", "")).upper()
            eng = by_name.get(ENG.get(name, name))
            if eng is None:
                break
            data_ = {"x": int(a.get("col", 0)), "y": int(a.get("row", 0))} if name == "MOUSE" else None
            g0 = grid(obs); obs, *_ = env.step(eng, data_); executed += 1; step_counter["n"] += 1
            g1 = grid(obs); changed = changed or (g0 != g1)
            history.append((name, g1, level(obs)))
            if getattr(getattr(obs, "state", None), "name", "") == "GAME_OVER":
                over = True; break
            if level(obs) > before_level:
                lvl_done = True; break
        res = {"executed": True, "executed_count": executed, "board_changed": changed, "level": level(obs),
               "level_completed": lvl_done, "game_over": over, "run_complete": False,
               "action_num": step_counter["n"], "state": getattr(getattr(obs, "state", None), "name", "")}
        return {"action_result": res, "state": state_payload(res)}

    code = PRELUDE + f"\nresult = goto({target[0]}, {target[1]})\nprint(result)\n"
    t0 = time.time()
    out = run_sandboxed_python(code=code, timeout_seconds=90, initial_state=state_payload(), action_handler=handler)
    dt = time.time() - t0
    res = out.get("result")
    err = out.get("error")
    return {"gid": gid[:4], "learned": True, "mover": model["mover"], "rules": {k_: v[1] for k_, v in model["rules"].items()},
            "start": [start[0], start[1]] if start else None, "target": list(target), "human_moves": m,
            "reached": bool(isinstance(res, dict) and res.get("reached")), "result": res, "error": err, "diag": diag, "k": k,
            "stdout": (out.get("stdout") or "")[-300:], "seconds": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games"); ap.add_argument("--k", type=int, default=12); ap.add_argument("--m", type=int, default=8)
    args = ap.parse_args()
    rows = []
    for p in sorted(REPLAYS.glob("*.json")):
        data = json.loads(p.read_text()); gid = data["game_id"]
        if args.games and gid[:4] not in set(args.games.split(",")):
            continue
        try:
            r = run_game(gid, data, args.k, args.m, print)
        except Exception as e:  # noqa: BLE001
            import traceback; r = {"gid": gid[:4], "error": f"{type(e).__name__}: {str(e)[:160]}"}; traceback.print_exc(limit=2)
        rows.append(r)
        flag = "REACHED" if r.get("reached") else ("skip" if r.get("skip") else ("no-learn" if r.get("learned") is False else "failed"))
        d = r.get("diag") or {}
        print(f"{r['gid']}: {flag:8s} mover {r.get('mover')} rules {r.get('rules')} k={r.get('k')} start {r.get('start')} -> target {r.get('target')} "
              f"| floor {d.get('floor')} pass {d.get('passable')} block {d.get('blocking')} bodies {d.get('n_bodies')} | offline plan strict {d.get('plan_strict')} opt {d.get('plan_optimistic')} "
              f"| {r.get('result') if not r.get('reached') else 'actions ' + str(r['result'].get('actions_taken')) + ' vs human ' + str(r.get('human_moves'))} "
              f"| err {r.get('error') or ''} | skip {r.get('skip') or ''}")
    tested = [r for r in rows if r.get("target")]
    print(f"\nSUMMARY: {sum(1 for r in tested if r.get('reached'))}/{len(tested)} targets reached; "
          f"{sum(1 for r in rows if r.get('learned') is False)} games with no learnable rules; {sum(1 for r in rows if r.get('skip'))} skipped")


if __name__ == "__main__":
    main()
