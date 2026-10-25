"""Measure the agent the way the competition runs it, and estimate the score.

With one Kaggle submission per day and an ~8-hour rerun, a leaderboard number
costs a full day. This is the local stand-in — but only if it reproduces the
conditions that actually decide the score.

An earlier version of this script did not. It drove one agent at a time,
in-process, in a single thread. The competition does none of those things:
`Swarm` starts one thread per game, all at once, all talking to a Flask
gateway over HTTP. Three faults lived in that gap, and every one of them was
invisible to the in-process path while costing real leaderboard points:

  * an environment returning None raised, and because Swarm's threads are
    daemons, that ended the game silently — 16 of 25 games died this way
  * an exception anywhere in the policy did the same
  * once those were caught, a permanently-jammed game livelocked instead

So this now starts a real gateway, runs every game concurrently against it,
and reports the health counters alongside the score. A run where threads die
or frames drop is not comparable to one where they don't, and that has to be
visible.

    .venv/bin/python scripts/evaluate.py                     # all games, 3 seeds, 80 actions
    .venv/bin/python scripts/evaluate.py --budget 2000 --seeds 2
    .venv/bin/python scripts/evaluate.py --seconds 120       # wall-clock, as the rerun is
    .venv/bin/python scripts/evaluate.py --games ls20,vc33
    .venv/bin/python scripts/evaluate.py --agent /tmp/old.py # compare a past version

The headline number is `est_score`: the sum over games of
`levels_completed / win_levels`, averaged over seeds. That form is a guess at
the leaderboard's metric, chosen because leaderboard values exceed 1.0 (top is
1.86), which rules out a plain mean of per-game fractions. It has never
matched the leaderboard in absolute terms — measured ratios of 0.30, 0.45,
0.19 — so use it to rank two versions, never to predict a score. See
`SUBMISSIONS.md` for what actually scored what.

`--out results.json` writes the full per-game table, so two runs can be
diffed rather than eyeballed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
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


# The games that have ever finished a level locally. Everything else scores
# exactly zero, so a mean over all 25 dilutes any change by 3-4x and hides a
# regression on the games that carry the score inside eighteen zeros that
# cannot move. Judge changes here first, then confirm on the full set.
#
# Ordered by what they cost per level, because that is what the score is:
#   r11l  8, 19        lp85  31, 60        vc33  50, 86, 276, 600
#   sp80  26, 43       m0r0  136, 298      ar25  109, 238, 567
#   ls20  526          ft09  9,063         s5i5  949
# The last three are included because they do score, but note ft09 and s5i5
# only ever managed it at a 20,000-action budget and at a cost that rounds to
# zero points — breadth bought that slowly is worth nothing.
SCORING_SET = ["r11l", "lp85", "vc33", "sp80", "m0r0", "ar25", "ls20"]


class _ThreadedWSGIServer(ThreadingMixIn, WSGIServer):
    """The competition gateway serves every agent at once. A single-threaded
    server would queue them behind each other and manufacture failures the
    real environment may not have."""

    daemon_threads = True


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_gateway(port: int) -> None:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    app, _ = create_app(arcade, competition_mode=True)
    server = make_server("127.0.0.1", port, app, server_class=_ThreadedWSGIServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def arc_score(level_actions: list[int], win_levels: int, baseline: int) -> float:
    """The competition's own per-game score, from `arc_agi/scorecard.py`.

        level score  = min((baseline_actions / actions_taken) ** 2 * 100, 115)
                       for a completed level, 0 otherwise
        game score   = sum(level_score * level_index) / sum(all level_indices)
        final score  = mean of game scores          (0-100)

    Two things in that are easy to miss and change what is worth building.

    **Actions are the score, squared — not a tiebreaker.** Finishing a level
    in twice the baseline scores 25 of a possible 100; in ten times, 1.0; in a
    hundred times, 0.01. This agent takes hundreds to thousands of actions per
    level, so the levels it does finish are already worth almost nothing, and
    "complete more levels" has been the wrong objective.

    That also explains the leaderboard's scale. ARC's published figure for the
    best preview agent was 12.58% of human action-efficiency, and
    0.1258**2 * 100 = 1.58 — squarely inside the 1.46-1.86 pack at the top.
    Our own 0.25 works out at about 5% of human efficiency. The gap is not six
    times more levels; it is roughly 2.7x fewer actions on the levels already
    being finished, and because the term is squared, halving actions
    quadruples the score.

    **Uncompleted levels still carry their weight in the denominator**, so the
    divisor is the whole game (`win_levels`), and later levels count for more
    than early ones.

    `baseline` is the one thing we cannot know: `api.py` strips
    `baseline_actions` from what the local gateway serves, and every local game
    reports an empty list. **It is a constant multiplier only while the 115 cap
    does not bind**, and the cap is not hypothetical — at baseline 100, six of
    sixteen levels won in a 25-game run were capped, and the whole score moved
    0.150 -> 0.997 across baselines 10 to 200. A capped level is one the proxy
    has stopped measuring: further efficiency there earns nothing here even
    though it would in the real thing. The default is deliberately low so this
    stays rare, and `main` prints the count whenever any level caps.
    """
    if win_levels <= 0:
        return 0.0
    total_weight = win_levels * (win_levels + 1) // 2
    total = 0.0
    for index, actions in enumerate(level_actions[:win_levels], start=1):
        if actions > 0:
            total += min((baseline / actions) ** 2 * 100, 115.0) * index
    return total / total_weight


def load_agent_class(path: Path):
    spec = importlib.util.spec_from_file_location("eval_agent_module", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve __module__.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise SystemExit(f"{path} must define a class named `MyAgent`")
    return module.MyAgent


def play_round(agent_cls, arc, game_ids, seed, budget, seconds, port):
    """One seed: build every game, then play them all at once, as `Swarm` does.

    Each round gets its own gateway (see `main`). Sharing one across rounds
    leaves the previous round's sessions open, and the next round's `make()`
    collides with them: a three-game, two-seed run dropped 900 of 1,800 frames
    and forced 297 resets, none of which was the agent's doing.

    The construction/play split matters and is not cosmetic. `Swarm.main`
    (agents/swarm.py:76-95) calls `arc.make()` for all 25 games sequentially on
    the main thread and only then starts the threads. An earlier version of
    this function called `make()` *inside* each worker instead, so 25 startup
    RESETs raced. That manufactured a failure the competition does not have:
    one game per run -- tn36 in one, bp35 and vc33 in the next -- came back
    with `dropped == actions`, meaning it never completed a single step, and
    produced practically all of the run's dropped frames and every one of its
    forced resets. Build first, play second, or the harness invents its own
    casualties and hides the real ones underneath them.
    """
    results: dict[str, dict] = {}
    lock = threading.Lock()

    built: dict[str, object] = {}
    build_errors: dict[str, str] = {}
    for game_id in game_ids:
        try:
            env = arc.make(game_id)
            agent = agent_cls(
                card_id="eval", game_id=game_id, agent_name=f"eval.{game_id}",
                ROOT_URL=f"http://127.0.0.1:{port}", record=False,
                arc_env=env, tags=["eval"],
            )
            # Fixed seed so a rerun of the same agent gives the same numbers;
            # the agent normally seeds itself from the wall clock.
            agent._rng = random.Random(seed)
            built[game_id] = (env, agent)
        except Exception as e:
            build_errors[game_id] = f"{type(e).__name__}: {e}"

    def run(game_id: str) -> None:
        record = {
            "game": game_id, "seed": seed, "levels": 0, "win_levels": 0,
            "actions": 0, "dropped": 0, "errors": 0, "resets": 0,
            "reopens": 0, "level_actions": [], "error": build_errors.get(game_id),
        }
        if record["error"]:
            with lock:
                results[game_id] = record
            return
        try:
            env, agent = built[game_id]
            deadline = time.monotonic() + seconds if seconds else None
            taken = 0
            # Where the current level started, and how many were done by then.
            level_start = 0
            level_mark = agent.frames[-1].levels_completed
            while True:
                if deadline is not None:
                    if time.monotonic() >= deadline:
                        break
                elif taken >= budget:
                    break
                if agent.is_done(agent.frames, agent.frames[-1]):
                    break
                # Mirror `Agent.main` (agents/agent.py:76-87) exactly, including
                # the `observation_space` read it passes to `choose_action`.
                # Skipping that read is not a shortcut: it is a *stale* frame
                # on a jammed game, and letting the agent see it changed the
                # agent's own behaviour. With the read omitted, a game whose
                # steps were being refused produced 9 forced resets over 30
                # iterations; with it, the same jam produced 0, because the
                # non-None stale frame kept clearing the failure counter. This
                # harness reported an escape working that could not fire in the
                # competition.
                latest = agent._convert_raw_frame_data(env.observation_space)
                action = agent.choose_action(agent.frames, latest)
                frame = agent.take_action(action)
                if frame is not None:
                    agent.append_frame(frame)
                    record["levels"] = max(record["levels"], frame.levels_completed)
                    record["win_levels"] = max(
                        record["win_levels"], frame.win_levels or 0
                    )
                    # How many actions this level cost. The competition scores
                    # on exactly this, squared — see `arc_score`.
                    done = frame.levels_completed
                    if done > level_mark:
                        span = taken + 1 - level_start
                        gained = done - level_mark
                        # More than one level in a single action is not
                        # expected; split the span rather than lose it.
                        each = max(span // gained, 1)
                        record["level_actions"].extend([each] * gained)
                        level_start = taken + 1
                    level_mark = done
                taken += 1
            record["actions"] = taken
            record["dropped"] = getattr(agent, "_dropped_frames", 0)
            record["errors"] = getattr(agent, "_policy_errors", 0)
            record["resets"] = getattr(agent, "_forced_resets", 0)
            record["reopens"] = getattr(agent, "_session_reopens", 0)
        except Exception as e:
            # In the real run a dead thread is silent; here it is a headline.
            record["error"] = f"{type(e).__name__}: {e}"
        with lock:
            results[game_id] = record

    threads = [threading.Thread(target=run, args=(g,)) for g in game_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return list(results.values())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", default="all",
                   help="Comma-separated game ids, 'all' (default), or "
                        "'scoring' for the games that have ever finished a "
                        "level here — see SCORING_SET.")
    p.add_argument("--seeds", type=int, default=3,
                   help="Seeds per game (default 3). Results vary a lot per seed.")
    p.add_argument("--budget", type=int, default=80,
                   help="Actions per game (default 80). Ignored when --seconds is set.")
    p.add_argument("--seconds", type=float, default=None,
                   help="Wall-clock seconds per game instead of an action budget — "
                        "closer to the rerun, but less reproducible.")
    p.add_argument("--agent", default=str(ROOT / "agent" / "my_agent.py"),
                   help="Agent file to evaluate (default agent/my_agent.py).")
    p.add_argument("--baseline", type=int, default=25,
                   help="Assumed human baseline actions per level (default 25). "
                        "The real values are stripped from what the gateway "
                        "serves. Kept low so the 115-point cap rarely binds — "
                        "above the cap the proxy stops rewarding efficiency, "
                        "which is the one thing it exists to measure.")
    p.add_argument("--out", default=None, help="Write the full table as JSON here.")
    args = p.parse_args()

    # The engine logs a line per game load and the gateway one per request.
    logging.disable(logging.INFO)
    for noisy in ("werkzeug", "arc_agi", "arc_agi.scorecard", "arc_agi.api"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    agent_cls = load_agent_class(Path(args.agent))

    import os
    os.environ.setdefault("ARC_API_KEY", "eval-key")

    def fresh_gateway():
        port = free_port()
        start_gateway(port)
        time.sleep(1.5)
        os.environ["ARC_BASE_URL"] = f"http://127.0.0.1:{port}"
        return port, arc_agi.Arcade(operation_mode=OperationMode.ONLINE)

    port, arc = fresh_gateway()
    all_ids = [e.game_id.split("-")[0] for e in arc.get_environments()]
    if args.games == "all":
        game_ids = all_ids
    elif args.games == "scoring":
        game_ids = [g for g in all_ids if g in SCORING_SET]
    else:
        wanted = {g.strip().split("-")[0] for g in args.games.split(",")}
        game_ids = [g for g in all_ids if g in wanted]
        missing = wanted - set(game_ids)
        if missing:
            raise SystemExit(f"Unknown game id(s): {sorted(missing)}")

    limit = (f"{args.seconds:g}s" if args.seconds else f"{args.budget} actions")
    print(f"agent   {args.agent}")
    print(f"config  {len(game_ids)} games concurrent x {args.seeds} seeds x {limit}")
    print(f"        via gateway on 127.0.0.1:{port}\n")

    rows: list[dict] = []
    started = time.time()
    for seed in range(args.seeds):
        if seed:
            # A fresh gateway per round; see play_round for why sharing one
            # corrupts every round after the first.
            port, arc = fresh_gateway()
        rows.extend(
            play_round(agent_cls, arc, game_ids, seed,
                       args.budget, args.seconds, port)
        )
    elapsed = time.time() - started

    by_game: dict[str, list[dict]] = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(r)

    print(f"{'game':<8}{'levels/seed':>16}{'win_lv':>8}{'actions/level':>15}"
          f"{'arc_score':>11}{'fraction':>10}")
    print("-" * 68)
    est = 0.0
    arc_total = 0.0
    all_level_actions: list[int] = []
    for game_id in game_ids:
        rs = [r for r in by_game.get(game_id, []) if not r["error"]]
        if not rs:
            print(f"{game_id:<8}{'ERROR':>16}")
            continue
        levels = [r["levels"] for r in rs]
        win = max(r["win_levels"] for r in rs)
        mean = sum(levels) / len(levels)
        frac = mean / win if win else 0.0
        est += frac
        game_arc = sum(
            arc_score(r["level_actions"], win, args.baseline) for r in rs
        ) / len(rs)
        arc_total += game_arc
        spent = [a for r in rs for a in r["level_actions"]]
        all_level_actions.extend(spent)
        per = f"{sum(spent) / len(spent):,.0f}" if spent else "-"
        print(f"{game_id:<8}{str(levels):>16}{win:>8}{per:>15}"
              f"{game_arc:>11.3f}{frac:>10.3f}{'  <-' if mean else ''}")

    ok = [r for r in rows if not r["error"]]
    died = [r for r in rows if r["error"]]
    actions = sum(r["actions"] for r in ok)
    solved = sum(1 for rs in by_game.values() if any(r["levels"] for r in rs))
    arc_mean = arc_total / len(by_game) if by_game else 0.0

    print("-" * 68)
    print(f"games with any progress : {solved}/{len(by_game)}")
    print(f"total actions spent     : {actions:,}  ({actions / max(elapsed, 1):,.0f}/sec)")
    if all_level_actions:
        mean_cost = sum(all_level_actions) / len(all_level_actions)
        print(f"actions per level won   : {mean_cost:,.0f}  "
              f"(best {min(all_level_actions):,}, "
              f"{len(all_level_actions)} levels won)")
    capped = sum(1 for a in all_level_actions
                 if (args.baseline / a) ** 2 * 100 > 115.0)
    print(f"arc_score               : {arc_mean:.3f}   "
          f"<- the competition's formula (baseline={args.baseline})")
    if capped:
        # Above the cap the proxy no longer rewards being faster, so a run
        # with many capped levels is not measuring what it claims to.
        print(f"  warning: {capped} of {len(all_level_actions)} levels hit the "
              f"115 cap — lower --baseline for a comparison that still moves")
    print(f"est_score               : {est:.3f}   (old level-count proxy)")
    print("\nhealth (invisible to the old in-process harness):")
    print(f"  threads that died     : {len(died)}")
    print(f"  frames dropped        : {sum(r['dropped'] for r in ok)}")
    print(f"  policy errors         : {sum(r['errors'] for r in ok)}")
    print(f"  forced resets         : {sum(r['resets'] for r in ok)}")
    print(f"  sessions re-opened    : {sum(r['reopens'] for r in ok)}")
    # A game that loses its startup reset produces almost all of a run's drops
    # and resets by itself, so the totals above hide it. Name the worst.
    worst = max(ok, key=lambda r: r["dropped"], default=None)
    if worst and worst["dropped"] > 4 * (sum(r["dropped"] for r in ok) / max(len(ok), 1)):
        print(f"    concentrated in {worst['game']} seed {worst['seed']}: "
              f"{worst['dropped']} dropped, {worst['resets']} resets, "
              f"{worst['reopens']} re-opens")
    for r in died[:5]:
        print(f"    {r['game']} seed {r['seed']}: {r['error']}")

    print("\n(est_score ranks two versions; it has never matched the leaderboard\n"
          " in absolute terms — see SUBMISSIONS.md.)")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"agent": args.agent, "budget": args.budget, "seconds": args.seconds,
             "seeds": args.seeds, "est_score": est, "rows": rows}, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
