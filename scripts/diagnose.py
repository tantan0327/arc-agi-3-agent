"""Ask why each game scores what it scores, by reading the agent's own models.

`evaluate.py` answers "how many levels" and nothing else, which is the right
question for accepting a change and the wrong one for deciding what to build.
Fifteen of twenty-five games score zero, and they do not score zero for the
same reason: some never find an action that does anything, some find plenty
and never work out what the directional keys mean, some understand movement
and have nowhere to go. Those want three different pieces of work, and the
level count cannot tell them apart.

So this plays each game exactly as `evaluate.py` does -- built sequentially,
run concurrently, through a real gateway, mirroring `Agent.main`'s loop -- and
then opens up the agent and reports what each layer managed to learn:

    effect      did any action ever change the board, and how often
    states      how many distinct states the abstraction produced
    motion      was an avatar found, and a direction scheme learned for it
    contact     were hazard or goal colours identified
    goal        were any colour-pair predicates confirmed
    milestone   did any transition ever raise the score

A game with `effect 0%` has a perception or targeting problem. A game with
high effect and no motion scheme has a causal-model problem. A game with a
scheme and no goal has a goal-inference problem. That is the split this exists
to make.

    .venv/bin/python scripts/diagnose.py
    .venv/bin/python scripts/diagnose.py --budget 2000 --games ls20,wa30
    .venv/bin/python scripts/diagnose.py --out /tmp/diag.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import random
import socket
import sys
import threading
import time
from pathlib import Path
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, make_server

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arc_agi.server import create_app  # noqa: E402


class _ThreadedWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


def start_gateway() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = int(s.getsockname()[1])
    arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    app, _ = create_app(arcade, competition_mode=True)
    server = make_server("127.0.0.1", port, app, server_class=_ThreadedWSGIServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(1.5)
    return port


def load_agent_class(path: Path):
    spec = importlib.util.spec_from_file_location("diag_agent_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MyAgent


def kind_of(actions: list[int]) -> str:
    """Click-only, move-only, or mixed — the split that decides which of the
    agent's layers even applies to a game."""
    has_click = 6 in actions
    has_move = bool(set(actions) & {1, 2, 3, 4, 5})
    if has_click and has_move:
        return "mixed"
    return "click" if has_click else "move"


def inspect(agent, env) -> dict:
    """Read the learned models. Every accessor is wrapped because this runs
    against whatever the agent happens to be today, and a diagnostic that
    crashes on a renamed field is worse than one that reports a blank."""
    def safe(fn, default=None):
        try:
            return fn()
        except Exception:
            return default

    model = getattr(agent, "_model", None)
    transitions = getattr(model, "transitions", {}) or {}
    tries = sum(t.tries for t in transitions.values())
    did_something = sum(t.tries for t in transitions.values() if t.frame_changed_ever)

    motion = getattr(agent, "_motion", None)
    avatar = safe(lambda: motion.avatar_id())
    scheme = safe(lambda: motion.scheme_for_object(avatar), {}) if avatar is not None else {}

    contact = getattr(agent, "_contact", None)
    goal = getattr(agent, "_goal", None)

    return {
        "levels": agent.frames[-1].levels_completed,
        "win_levels": agent.frames[-1].win_levels or 0,
        "state": agent.frames[-1].state.name,
        "kind": kind_of(list(agent.frames[-1].available_actions or [])),
        "actions_seen": sorted(agent.frames[-1].available_actions or []),
        # Perception / targeting: did anything the agent did ever move a pixel.
        "effect_pct": round(100.0 * did_something / tries, 1) if tries else 0.0,
        # State abstraction: too many states and nothing generalises, too few
        # and the agent cannot tell situations apart.
        "states": len(getattr(model, "adjacency", {}) or {}),
        "milestones": len(getattr(model, "milestone_states", set()) or set()),
        # Causal motion layer.
        "avatar": avatar,
        "scheme": {k: list(v) for k, v in (scheme or {}).items()},
        "noise_colors": sorted(safe(lambda: motion.noise_colors(), set()) or set()),
        # Goal direction.
        "danger_colors": sorted(safe(lambda: contact.danger_colors(), set()) or set()),
        "goal_colors": sorted(safe(lambda: contact.goal_colors(), set()) or set()),
        "predicates": len(safe(lambda: goal.confirmed_predicates(), set()) or set()),
        "dropped": getattr(agent, "_dropped_frames", 0),
        "errors": getattr(agent, "_policy_errors", 0),
        "resets": getattr(agent, "_forced_resets", 0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", default="all")
    p.add_argument("--budget", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--agent", default=str(ROOT / "agent" / "my_agent.py"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    logging.disable(logging.INFO)
    for noisy in ("werkzeug", "arc_agi", "arc_agi.scorecard", "arc_agi.api"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    agent_cls = load_agent_class(Path(args.agent))
    os.environ.setdefault("ARC_API_KEY", "diag-key")
    port = start_gateway()
    os.environ["ARC_BASE_URL"] = f"http://127.0.0.1:{port}"
    arc = arc_agi.Arcade(operation_mode=OperationMode.ONLINE)

    all_ids = [e.game_id.split("-")[0] for e in arc.get_environments()]
    if args.games == "all":
        game_ids = all_ids
    else:
        wanted = {g.strip().split("-")[0] for g in args.games.split(",")}
        game_ids = [g for g in all_ids if g in wanted]

    # Build first, play second — as Swarm does. See play_round in evaluate.py.
    built = {}
    for game_id in game_ids:
        env = arc.make(game_id)
        agent = agent_cls(
            card_id="diag", game_id=game_id, agent_name=f"diag.{game_id}",
            ROOT_URL=f"http://127.0.0.1:{port}", record=False,
            arc_env=env, tags=["diag"],
        )
        agent._rng = random.Random(args.seed)
        built[game_id] = (env, agent)

    print(f"agent   {args.agent}")
    print(f"config  {len(game_ids)} games concurrent x {args.budget} actions "
          f"(seed {args.seed})\n")

    rows: dict[str, dict] = {}
    lock = threading.Lock()

    def run(game_id: str) -> None:
        env, agent = built[game_id]
        try:
            for _ in range(args.budget):
                if agent.is_done(agent.frames, agent.frames[-1]):
                    break
                latest = agent._convert_raw_frame_data(env.observation_space)
                action = agent.choose_action(agent.frames, latest)
                frame = agent.take_action(action)
                if frame is not None:
                    agent.append_frame(frame)
            row = inspect(agent, env)
        except Exception as e:
            row = {"error": f"{type(e).__name__}: {e}"}
        row["game"] = game_id
        with lock:
            rows[game_id] = row

    threads = [threading.Thread(target=run, args=(g,)) for g in game_ids]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - started

    ordered = sorted(rows.values(),
                     key=lambda r: (r.get("kind", ""), -r.get("levels", 0),
                                    r.get("game", "")))

    print(f"{'game':<7}{'kind':<7}{'lv':>3}{'/win':>5}{'effect':>8}{'states':>8}"
          f"{'avatar':>8}{'scheme':>8}{'danger':>8}{'goal':>7}{'pred':>6}")
    print("-" * 75)
    for r in ordered:
        if r.get("error"):
            print(f"{r['game']:<7}ERROR  {r['error'][:55]}")
            continue
        print(f"{r['game']:<7}{r['kind']:<7}{r['levels']:>3}{r['win_levels']:>5}"
              f"{r['effect_pct']:>7.1f}%{r['states']:>8}"
              f"{('yes' if r['avatar'] is not None else '-'):>8}"
              f"{(len(r['scheme']) or '-'):>8}"
              f"{(len(r['danger_colors']) or '-'):>8}"
              f"{(len(r['goal_colors']) or '-'):>7}"
              f"{(r['predicates'] or '-'):>6}")

    print("-" * 75)
    print(f"elapsed {elapsed:.0f}s")

    # The point of the table: sort the zeros into the work they imply.
    zeros = [r for r in ordered if not r.get("error") and not r["levels"]]
    print(f"\n{len(zeros)} games at zero, by what is missing:")
    # Click-only games have no avatar and no direction scheme, so the motion
    # tests below do not apply to them and must not be allowed to answer for
    # them. Order matters: a game that barely moves the board has a perception
    # problem whatever else is also true of it.
    buckets: dict[str, list[str]] = {}
    for r in zeros:
        if r["effect_pct"] < 20:
            b = (f"barely anything it does changes the board "
                 f"(perception/targeting)")
        elif r["kind"] == "click":
            b = ("clicks land but nothing has ever scored, so there is no "
                 "success to learn a goal from (goal inference)")
        elif not r["scheme"]:
            b = "acts on the board but never learns what the moves mean (causal model)"
        elif not r["goal_colors"]:
            b = "knows how to move, has no destination (goal inference)"
        else:
            b = "acts and understands, still cannot finish a level (planning/depth)"
        buckets.setdefault(b, []).append(r["game"])
    for b, gs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(gs):>2}  {b}")
        print(f"      {', '.join(sorted(gs))}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"agent": args.agent, "budget": args.budget, "seed": args.seed,
             "rows": ordered}, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
