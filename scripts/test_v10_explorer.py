"""Offline v10 test against the REAL engine: does harness-forced exploration
over an abstraction reach levels a reactive policy does not?

No GPU, no LLM. The reactive policy is a weak stand-in (random valid action,
occasional click on a small blob). Arm A: reactive only. Arm B: the v10
explorer with an abstraction supplied -- the GP-B hand-written references on
their 3 games (standing in for an authored one), the generic object-set rung
elsewhere -- falling back to the same reactive policy when it has no plan.
Same seeds, same action budget. Levels completed is the score that matters
(per-level score is capped, only reach counts).

    python scripts/test_v10_explorer.py [--budget 300] [--games m0r0,tr87,s5i5]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor/taaf-kaggle-source/src/re-arc-3"))

from duck_delta.v10_explorer import V10Explorer, click_candidates, blobs_for_authoring  # noqa: E402
from gpb_reference_abstractions import ABSTRACTIONS  # noqa: E402

E2M = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE", "ACTION6": "MOUSE", "RESET": "RESET"}
M2E = {v: k for k, v in E2M.items()}


def generic_object_set(board):
    b = blobs_for_authoring(board)
    return tuple(sorted((c, len(v)) for c, v in b.items()))


def run_game(gid, budget, seed, use_explorer, log=None):
    from re_arc import EnvSampler
    from arcengine import GameAction
    rng = random.Random(seed)
    env = EnvSampler(include=[gid]).make(gid)
    obs = env.reset()
    by_name = {m.name: m for m in GameAction}

    def board(o):
        return [list(map(int, r)) for r in o.frame[-1]]

    def level(o):
        return int(getattr(o, "levels_completed", 0) or 0) + 1

    def valid(o):
        av = getattr(o, "available_actions", None)
        names = [E2M.get(getattr(a, "name", str(a)), None) for a in (av or [])]
        names = [n for n in names if n]
        return names or ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]

    ex = None
    if use_explorer:
        ex = V10Explorer(chat=None, log=log, warmup=15)
        ex.set_abstraction(ABSTRACTIONS.get(gid[:4], generic_object_set))
        ex.seed_initial(board(obs), level(obs))
    history = []
    actions = 0; max_level = level(obs); completed = 0

    def do(spec):
        nonlocal obs, actions, completed, max_level
        name = spec["action"]; eng = by_name.get(M2E.get(name, name))
        if eng is None:
            return
        data = {"x": int(spec["col"]), "y": int(spec["row"])} if name == "MOUSE" else None
        obs, _r, _d, _i = env.step(eng, data)
        actions += 1
        disp = f"MOUSE(row={spec['row']}, col={spec['col']})" if name == "MOUSE" else name
        history.append((disp, board(obs), level(obs)))
        completed = int(getattr(obs, "levels_completed", 0) or 0)
        max_level = max(max_level, level(obs))
        st = getattr(getattr(obs, "state", None), "name", "")
        if st == "GAME_OVER":
            obs, _r, _d, _i = env.step(by_name["RESET"], None)
            history.append(("RESET", board(obs), level(obs)))

    while actions < budget:
        st = getattr(getattr(obs, "state", None), "name", "")
        if st == "WIN":
            break
        plan = None
        if ex is not None:
            plan = ex.propose(board(obs), level(obs), history, valid(obs))
        if plan:
            for spec in plan[: budget - actions]:
                do(spec)
            continue
        v = [n for n in valid(obs) if n != "RESET"]
        if "MOUSE" in v and rng.random() < 0.3:
            cc = click_candidates(board(obs), 16)
            if cc:
                cy, cx = rng.choice(cc); do({"action": "MOUSE", "row": cy, "col": cx}); continue
        v2 = [n for n in v if n != "MOUSE"] or v
        do({"action": rng.choice(v2)})
    return completed, actions, (ex.stats if ex else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--games")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    import json
    replays = ROOT / "vendor/taaf-kaggle-source/src/re-arc-3/re_arc/dsl/official_human_replays"
    gids = [json.loads(p.read_text())["game_id"] for p in sorted(replays.glob("*.json"))]
    if args.games:
        want = set(args.games.split(",")); gids = [g for g in gids if g[:4] in want]
    tot = {"A": 0, "B": 0}
    print(f"{'game':6} {'ref?':5} {'A:react':>8} {'B:v10':>7}  stats")
    for gid in gids:
        ra = rb = 0; stats = None
        for s in range(args.seeds):
            try:
                a, _, _ = run_game(gid, args.budget, s, False)
                b, _, st = run_game(gid, args.budget, s, True, log=(print if args.verbose else None))
            except Exception as e:  # noqa: BLE001
                print(f"{gid[:4]:6} error {type(e).__name__}: {str(e)[:80]}"); break
            ra += a; rb += b; stats = st
        tot["A"] += ra; tot["B"] += rb
        ref = "ref" if gid[:4] in ABSTRACTIONS else ""
        print(f"{gid[:4]:6} {ref:5} {ra:8d} {rb:7d}  {dict(stats) if stats else ''}")
    print(f"\nTOTAL levels over {args.seeds} seeds: reactive {tot['A']}  v10 {tot['B']}")


if __name__ == "__main__":
    main()
