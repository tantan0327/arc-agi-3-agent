"""Where the actions go, now that actions are the score.

The competition scores a completed level at `min((baseline/actions)**2*100, 115)`
and a level never completed at zero (`arc_agi/scorecard.py`). Actions are not a
tiebreaker; they are the whole quantity, squared. This agent spends hundreds to
thousands of them per level, so the levels it does finish score near nothing —
and before deciding what to cut, it is worth knowing what is actually being
bought.

Three questions, and none of them is answered by a level count:

  by mode      what kind of decision each action came from. "untried" is
               genuine novelty-seeking, "known"/"plan" are the world model
               paying off, "navigate" is the avatar being steered, and
               "random" is a coin flip with no information behind it. A run
               dominated by untried+random is one where nothing is being
               learned that the agent can act on.

  by phase     spend before the first level completes, versus after. Level 1
               carries weight 1 of a 7-level game's 28, so exploring hard
               there is nearly free; doing it forever is not. This says which
               one is happening.

  on dying     actions spent on attempts that ended in GAME_OVER. Those are
               charged to the level all the same, so a game that dies often
               pays for the level several times over.

    .venv/bin/python scripts/spend.py
    .venv/bin/python scripts/spend.py --games ls20,vc33 --budget 2000
"""
from __future__ import annotations

import argparse
import collections
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
from arcengine import GameState  # noqa: E402

MODES = ["navigate", "untried", "known", "plan", "random", "lookahead", "other"]


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
    spec = importlib.util.spec_from_file_location("spend_agent_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MyAgent


def play(agent, env, budget: int) -> dict:
    modes: collections.Counter[str] = collections.Counter()
    before_first: collections.Counter[str] = collections.Counter()
    # Per level: the mode mix and the deaths, so a level that costs ten times
    # the last one can be told apart from a level that is simply harder.
    # If knowledge carries forward, later levels should lean on known/navigate;
    # if every level is learned from scratch, they stay untried-dominated.
    per_level: list[dict] = []
    cur_modes: collections.Counter[str] = collections.Counter()
    cur_deaths = 0
    # Is the agent still holding on to its avatar?
    #
    # `_MotionModel.avatar_id` answers from persistent statistics, so it keeps
    # returning an id long after the tracker has lost the object — the id
    # survives a level change even when the board is replaced wholesale.
    # `_avatar_position` then falls back to the nearest object of the avatar's
    # colour, and on sp80 that fallback latched onto a decoration: the
    # reported position sat at (25, 33) for four straight steps while the real
    # avatar moved, navigation saw its moves fail, learned every direction as
    # a wall, and returned None from then on.
    #
    # Counted here rather than assumed, because it was seen on one game of
    # two and a fix is only worth shipping if it happens often. All three
    # reads are side-effect free.
    anchored = 0        # tracker still holds the avatar
    fallback = 0        # tracker lost it; running on the colour fallback
    frozen = 0          # ...and the fallback position has stopped moving
    no_avatar = 0
    last_resolved = None
    stuck_run = 0
    worst_stuck = 0
    level_actions: list[int] = []
    deaths = 0
    died_actions = 0        # actions on attempts that ended in GAME_OVER
    attempt_start = 0       # where the current attempt (since last death) began
    level_start = 0
    level_mark = agent.frames[-1].levels_completed

    for taken in range(budget):
        if agent.is_done(agent.frames, agent.frames[-1]):
            break
        latest = agent._convert_raw_frame_data(env.observation_space)
        action = agent.choose_action(agent.frames, latest)

        reasoning = getattr(action, "reasoning", None)
        mode = (reasoning or {}).get("mode") if isinstance(reasoning, dict) else None
        if action.name == "RESET":
            mode = mode or "reset"
        mode = mode if mode in MODES else ("reset" if mode == "reset" else "other")
        modes[mode] += 1
        cur_modes[mode] += 1
        if not level_actions:
            before_first[mode] += 1

        frame = agent.take_action(action)
        if frame is None:
            continue
        agent.append_frame(frame)

        # Read-only: position_of and avatar_id are pure, _last_avatar is the
        # position the agent itself settled on this step.
        aid = agent._motion.avatar_id()
        if aid is None:
            no_avatar += 1
            stuck_run = 0
        else:
            resolved = agent._last_avatar
            if agent._tracker.position_of(aid) is not None:
                anchored += 1
                stuck_run = 0
            else:
                fallback += 1
                # A move action that leaves the reported position untouched
                # means the fallback is following something that isn't the
                # avatar — or the avatar is against a wall, which is why this
                # is reported as a run length rather than a single step.
                if resolved is not None and resolved == last_resolved \
                        and mode in ("navigate", "untried", "known", "plan",
                                     "random"):
                    stuck_run += 1
                    frozen += 1
                    worst_stuck = max(worst_stuck, stuck_run)
                else:
                    stuck_run = 0
            last_resolved = resolved

        if frame.state is GameState.GAME_OVER:
            deaths += 1
            cur_deaths += 1
            # Everything since the last death was spent on an attempt that
            # ended in failure — and the level is charged for it regardless.
            died_actions += taken + 1 - attempt_start
            attempt_start = taken + 1

        done = frame.levels_completed
        if done > level_mark:
            span = taken + 1 - level_start
            for _ in range(done - level_mark):
                level_actions.append(max(span // (done - level_mark), 1))
            per_level.append({
                "level": len(level_actions),
                "actions": span,
                "deaths": cur_deaths,
                "modes": dict(cur_modes),
            })
            cur_modes = collections.Counter()
            cur_deaths = 0
            level_start = taken + 1
            attempt_start = taken + 1
        level_mark = done

    return {
        "levels": level_mark,
        "win_levels": agent.frames[-1].win_levels or 0,
        "level_actions": level_actions,
        "actions": sum(modes.values()),
        "modes": dict(modes),
        "before_first": dict(before_first),
        "deaths": deaths,
        "died_actions": died_actions,
        "per_level": per_level,
        "anchor": {
            "anchored": anchored, "fallback": fallback, "frozen": frozen,
            "no_avatar": no_avatar, "worst_stuck": worst_stuck,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", default="all")
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--agent", default=str(ROOT / "agent" / "my_agent.py"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    logging.disable(logging.INFO)
    for noisy in ("werkzeug", "arc_agi", "arc_agi.scorecard", "arc_agi.api"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    agent_cls = load_agent_class(Path(args.agent))
    os.environ.setdefault("ARC_API_KEY", "spend-key")
    port = start_gateway()
    os.environ["ARC_BASE_URL"] = f"http://127.0.0.1:{port}"
    arc = arc_agi.Arcade(operation_mode=OperationMode.ONLINE)

    all_ids = [e.game_id.split("-")[0] for e in arc.get_environments()]
    if args.games == "all":
        game_ids = all_ids
    else:
        wanted = {g.strip().split("-")[0] for g in args.games.split(",")}
        game_ids = [g for g in all_ids if g in wanted]

    # Build first, play second — as Swarm does; see evaluate.play_round.
    built = {}
    for game_id in game_ids:
        env = arc.make(game_id)
        agent = agent_cls(
            card_id="spend", game_id=game_id, agent_name=f"spend.{game_id}",
            ROOT_URL=f"http://127.0.0.1:{port}", record=False,
            arc_env=env, tags=["spend"],
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
            row = play(agent, env, args.budget)
        except Exception as e:
            row = {"error": f"{type(e).__name__}: {e}"}
        row["game"] = game_id
        with lock:
            rows[game_id] = row

    threads = [threading.Thread(target=run, args=(g,)) for g in game_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    shown = [m for m in ("navigate", "untried", "known", "plan", "random",
                         "lookahead", "reset", "other")]
    header = f"{'game':<7}{'lv':>3}{'actions':>9}"
    for m in shown:
        header += f"{m[:6]:>8}"
    header += f"{'deaths':>8}{'died%':>7}"
    print(header)
    print("-" * len(header))

    totals: collections.Counter[str] = collections.Counter()
    grand = 0
    for game_id in game_ids:
        r = rows.get(game_id, {})
        if r.get("error"):
            print(f"{game_id:<7}ERROR {r['error'][:60]}")
            continue
        n = r["actions"] or 1
        line = f"{game_id:<7}{r['levels']:>3}{r['actions']:>9}"
        for m in shown:
            v = r["modes"].get(m, 0)
            line += f"{(100.0 * v / n):>7.0f}%"
            totals[m] += v
        grand += r["actions"]
        line += f"{r['deaths']:>8}{100.0 * r['died_actions'] / n:>6.0f}%"
        print(line)

    print("-" * len(header))
    line = f"{'ALL':<7}{'':>3}{grand:>9}"
    for m in shown:
        line += f"{(100.0 * totals[m] / max(grand, 1)):>7.0f}%"
    print(line)

    won = [(g, a) for g in game_ids for a in rows.get(g, {}).get("level_actions", [])]
    print()
    if won:
        costs = [a for _, a in won]
        print(f"levels won              : {len(won)}")
        print(f"actions per level won   : mean {sum(costs)/len(costs):,.0f}, "
              f"best {min(costs):,}, worst {max(costs):,}")
    else:
        print("levels won              : 0")

    dead = sum(r.get("died_actions", 0) for r in rows.values())
    print(f"actions on fatal attempts: {dead:,} of {grand:,} "
          f"({100.0 * dead / max(grand, 1):.0f}%)")

    pre = collections.Counter()
    for r in rows.values():
        pre.update(r.get("before_first", {}))
    pre_total = sum(pre.values())
    print(f"spent before a first win : {pre_total:,} of {grand:,} "
          f"({100.0 * pre_total / max(grand, 1):.0f}%)")

    # Does a level cost more because it is harder, or because nothing learned
    # on the last one carried over? Group every level won by its index and
    # compare the mode mix. Knowledge that transfers should show up as later
    # levels leaning on known/navigate; levels learned from scratch stay
    # untried-dominated however deep they are.
    # How often is the agent steering something it has lost track of?
    have = [r for r in rows.values() if r.get("anchor", {}).get("anchored", 0)
            or r.get("anchor", {}).get("fallback", 0)]
    if have:
        print(f"\n{'game':<7}{'anchored':>10}{'fallback':>10}{'frozen':>9}"
              f"{'longest freeze':>16}")
        print("-" * 52)
        for r in sorted(have, key=lambda r: -r["anchor"]["fallback"]):
            a = r["anchor"]
            live = a["anchored"] + a["fallback"] or 1
            print(f"{r['game']:<7}{100.0*a['anchored']/live:>9.0f}%"
                  f"{100.0*a['fallback']/live:>9.0f}%"
                  f"{100.0*a['frozen']/live:>8.0f}%{a['worst_stuck']:>16}")
        tot_a = sum(r["anchor"]["anchored"] for r in have)
        tot_f = sum(r["anchor"]["fallback"] for r in have)
        tot_z = sum(r["anchor"]["frozen"] for r in have)
        live = tot_a + tot_f or 1
        print("-" * 52)
        print(f"{'ALL':<7}{100.0*tot_a/live:>9.0f}%{100.0*tot_f/live:>9.0f}%"
              f"{100.0*tot_z/live:>8.0f}%")

    depth: dict[int, list[dict]] = {}
    for r in rows.values():
        for lv in r.get("per_level", []):
            depth.setdefault(lv["level"], []).append(lv)
    if depth:
        print(f"\n{'level':<7}{'won':>5}{'actions':>9}{'deaths':>8}"
              f"{'untried':>9}{'navigate':>10}{'known':>8}{'plan':>7}")
        print("-" * 63)
        for level in sorted(depth):
            group = depth[level]
            acts = sum(g["actions"] for g in group) / len(group)
            dth = sum(g["deaths"] for g in group) / len(group)
            tot = sum(sum(g["modes"].values()) for g in group) or 1
            def pct(name: str) -> str:
                v = sum(g["modes"].get(name, 0) for g in group)
                return f"{100.0 * v / tot:.0f}%"
            print(f"{level:<7}{len(group):>5}{acts:>9,.0f}{dth:>8.1f}"
                  f"{pct('untried'):>9}{pct('navigate'):>10}"
                  f"{pct('known'):>8}{pct('plan'):>7}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
