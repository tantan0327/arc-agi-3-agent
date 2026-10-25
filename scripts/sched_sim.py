"""Offline, event-driven simulator of ARC-AGI-3's shared-server scheduler (v2, 2026-10-10).

Purpose: tune a slot-priority policy WITHOUT a GPU. The 9-hour contest run plays ~110 games
on one server with ARC3_MAX_ACTIVE_STREAMS=10 slots; a game keeps its slot until its
context is trimmed (58k generated tokens), then re-queues and the highest-priority waiter
takes the slot. That allocation is the only lever that acts solely in the hidden run, and
the field proved it moves the board (Affectify's D' retune: 31.54 vs Franzen's 27-31).

Mechanics reproduced from Franzen's harness (tool_agent._PriorityGate, priority_scheduler):
- fresh (never-started) games enter at 2,000,000 - queue_position: they beat any priced
  waiter (base), unless the policy prices them (D');
- at every pump the waiting snapshots are re-priced with the current clock (refresh queue);
- the tail bonus fades over the last ARC3_PRIORITY_TAIL_FADE_FRACTION of the run.

Cost model: an action's wall = its generation tokens / per-stream throughput, calibrated to
61 tok/s at 10 active streams (p25-p75 58-64 over 120 demo trajectories). The recorded
per-action `wallclock` is queue-inflated (~30x) and is never used.

Data: scratchpad/sched/traces.json - real trajectories (per action: tokens, clears), one
per (run, game). When a trace runs out before the game is finished, the game continues in
"hazard mode": each further action costs 1,800 tokens and clears the level with the
per-token clear hazard estimated for that game and level (own episodes pooled with the
population; games that never cleared anything get a small hazard). This is the honest
extrapolation of patience, the quantity D' tunes; longer traces (hardtrace2a/b) shrink it.

    python scripts/sched_sim.py --validate              # no contention: reproduces demo levels
    python scripts/sched_sim.py --calib                 # 25 games/10 slots/25 min vs the real run
    python scripts/sched_sim.py --compare base dprime fair --n 110 --hard-frac 0.5
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "scratchpad" / "sched" / "traces.json"

SLOTS = 10
DRAIN_TOKENS = 58 * 1024
WALL_BUDGET_S = 9 * 3600
PER_GAME_CAP_S = 532 * 60
HIDDEN_GAMES = 110
TPS10 = 61.0
STUCK_ACTION_TOKENS = 1800
FRESH_BASE = 2_000_000

EASY = ["ar25", "ft09", "lp85", "r11l", "re86", "sb26", "sc25", "tr87", "tu93", "vc33"]
HARD = ["bp35", "cd82", "cn04", "dc22", "g50t", "ka59", "lf52", "ls20", "m0r0", "s5i5", "sk48", "sp80", "su15", "tn36", "wa30"]


# ---- scoring -----------------------------------------------------------------------------
def game_score(levels_cleared, N, actions_per_level, base_apl):
    denom = sum(range(1, N + 1)) or 1
    num = 0.0
    for i in range(1, levels_cleared + 1):
        a = actions_per_level[i - 1] if i - 1 < len(actions_per_level) else None
        base = base_apl[i - 1] if base_apl and i - 1 < len(base_apl) else None
        eff = min((base / a) ** 2, 1.15) if (a and base) else 1.0
        num += i * eff
    return 100.0 * num / denom


def eta_curve(name):
    return {"linear": lambda k: 1.0 / max(1, k), "sqrt": lambda k: 1.0 / math.sqrt(max(1, k)),
            "batch": lambda k: 1.0 / (max(1, k) ** 0.35)}[name]


def norm_N(N):
    n = min(10, max(6, int(N or 10)))
    return 55.0 / (n * (n + 1) / 2)


# ---- hazard tables (continuation after a trace ends) --------------------------------------
HAZ_POP = {}          # level -> clears per token (population)
HAZ_GAME = {}         # gid -> (clears, tokens) over all its episodes
HAZ_PRIOR_TOKENS = 30000.0
HAZ_SCALE = 1.0       # global multiplier on the continuation hazard (calibrated to the board)
HAZ_HARD_SCALE = 1.0  # extra multiplier for the 15 hard games only (robustness: 0 = never clearable)
DIFFICULTY = 1.0      # token inflation of every trace action (hidden games are harder than public ones)


def build_hazards(traces):
    pop = {}
    per_game = {}
    for run in traces["runs"].values():
        for gid, g in run.items():
            lvl, toks = 1, 0
            c_g, t_g = per_game.get(gid, (0, 0))
            for tok, _w, clr in g["steps"]:
                toks += tok
                if clr:
                    c, t = pop.get(lvl, (0, 0)); pop[lvl] = (c + 1, t + toks)
                    c_g += 1; t_g += toks
                    lvl += 1; toks = 0
            if toks:
                c, t = pop.get(lvl, (0, 0)); pop[lvl] = (c, t + toks)
                t_g += toks
            per_game[gid] = (c_g, t_g)
    HAZ_POP.clear(); HAZ_GAME.clear()
    for lvl, (c, t) in pop.items():
        HAZ_POP[lvl] = c / max(1.0, t)
    HAZ_GAME.update(per_game)
    return dict(HAZ_POP), dict(HAZ_GAME)


def hazard_for(gid, level):
    pop = HAZ_POP.get(level) or HAZ_POP.get(max(HAZ_POP) if HAZ_POP else 1, 1e-5)
    base = gid.split("#")[0]
    c, t = HAZ_GAME.get(base, (0, 0))
    # own evidence pooled with the population prior (worth HAZ_PRIOR_TOKENS of spend)
    h = HAZ_SCALE * (c + pop * HAZ_PRIOR_TOKENS) / (t + HAZ_PRIOR_TOKENS)
    return h * HAZ_HARD_SCALE if base in HARD else h


# ---- game state --------------------------------------------------------------------------
class Game:
    __slots__ = ("gid", "N", "base_apl", "steps", "idx", "level", "act_on_level", "tok_on_level",
                 "tok_since_trim", "actions_per_level", "tok_total", "pace_tok", "pace_n",
                 "done", "slot_time", "started", "queue_pos", "rng", "continuing")

    def __init__(self, gid, trace, queue_pos, seed=0):
        self.gid = gid; self.N = int(trace.get("N") or 10); self.base_apl = trace.get("base_apl")
        self.steps = trace["steps"]; self.idx = 0; self.level = 1
        self.act_on_level = 0; self.tok_on_level = 0; self.tok_since_trim = 0
        self.actions_per_level = []; self.tok_total = 0; self.pace_tok = 0.0; self.pace_n = 0
        self.done = False; self.slot_time = 0.0; self.started = False; self.queue_pos = queue_pos
        self.rng = random.Random(seed * 7919 + queue_pos); self.continuing = False

    def pace(self):
        return self.pace_tok / self.pace_n if self.pace_n else 0.0

    def levels_cleared(self):
        return len(self.actions_per_level)

    def next_action(self, allow_continue):
        """-> (tokens, clears) of the next action, or None if the game is finished."""
        if self.idx < len(self.steps):
            tok, _w, clr = self.steps[self.idx]
            return int(tok * DIFFICULTY), bool(clr)
        if self.levels_cleared() >= self.N or not allow_continue:
            return None
        self.continuing = True
        p = hazard_for(self.gid, self.level) * STUCK_ACTION_TOKENS / DIFFICULTY
        return STUCK_ACTION_TOKENS, (self.rng.random() < p)

    def apply(self, tokens, clears, dt):
        if self.idx < len(self.steps):
            self.idx += 1
        self.act_on_level += 1; self.tok_on_level += tokens; self.tok_since_trim += tokens
        self.tok_total += tokens; self.slot_time += dt
        if clears:
            self.actions_per_level.append(self.act_on_level)
            self.pace_tok += self.tok_on_level; self.pace_n += 1
            self.level += 1; self.act_on_level = 0; self.tok_on_level = 0
            if self.levels_cleared() >= self.N:
                self.done = True
        trim = self.tok_since_trim >= DRAIN_TOKENS
        if trim:
            self.tok_since_trim = 0
        return trim

    def score(self):
        return game_score(self.levels_cleared(), self.N, self.actions_per_level, self.base_apl)


# ---- policies: value(g, phi) where phi = tail fraction in [0,1] (1 = no fade) -------------
def _tail_remaining(N, level):
    remaining = max(0, N - level)
    return (0.0, 5.0, 7.0, 8.0)[min(remaining, 3)]


def policy_base(g, phi, fade=0.2):
    """Franzen's contest configuration: normalize_score=1, tail_lookup=remaining (8/7/5/0),
    tail fade over the last 20%, pace disabled (1.0), human 25, scales 115/62000, weight .25.
    Fresh games jump the queue (2,000,000 - position)."""
    if not g.started:
        return FRESH_BASE - g.queue_pos
    level, a, t = g.level, g.act_on_level, g.tok_on_level
    eff = (25.0 / (25.0 + a)) ** 2
    A = level * eff * norm_N(g.N)
    tail = _tail_remaining(g.N, level) * phi
    chance = 0.25 * 0.5 ** ((a / 115.0) ** 2) + 0.75 * 0.5 ** ((t / 62000.0) ** 2)
    return max(1.0, (A + tail) * chance * 100.0)


def dprime_value(level, a, t, pace, N, phi):
    A = (1.0 + 0.5 * (level - 1)) * norm_N(N) * (300.0 / (300.0 + a)) ** 2.5
    if pace > 0:
        A *= min(4.0, max(0.25, (30000.0 / max(1.0, pace)) ** 0.4))
    rel = pace if pace > 0 else 30000.0
    scale = 225000.0 * (min(2.0, max(0.5, rel / 30000.0)) ** 0.5)
    C = 0.1 * max(0.1, 1.0 - a / 115.0) + 0.9 * max(0.1, 1.0 - t / scale)
    B = (0.0, 10.0, 14.0, 16.0)[min(max(0, N - level), 3)]
    return (A * C + B * phi) * 1000.0


def policy_dprime(g, phi):
    """The field's D' (benchmark to beat): priced fresh games, long patience, bigger tail."""
    if not g.started:
        return dprime_value(1, 0, 0.0, 0.0, g.N, phi)
    return dprime_value(g.level, g.act_on_level, g.tok_on_level, g.pace(), g.N, phi)


def policy_fair(g, phi):
    return 1.0 if g.started else FRESH_BASE - g.queue_pos


def policy_random(g, phi):
    return (hash(g.gid) % 100003) / 100003.0


# OURS: a parametric family that contains base and D' as points; searched offline.
OURS = {"level_pow": 0.5, "eff_const": 100.0, "eff_pow": 2.0, "patience_tok": 150000.0,
        "floor": 0.05, "b3": 12.0, "b2": 10.0, "b1": 7.0, "pace_pow": 0.5, "fresh_mult": 1.0,
        "stuck_pen": 0.5, "pace_ref": 20000.0}


def ours_value(level, a, t, pace, N, phi, p=OURS):
    A = (1.0 + p["level_pow"] * (level - 1)) * norm_N(N)
    A *= (p["eff_const"] / (p["eff_const"] + a)) ** p["eff_pow"]
    if pace > 0:
        A *= min(4.0, max(0.25, (p["pace_ref"] / max(1.0, pace)) ** p["pace_pow"]))
    scale = p["patience_tok"] * (min(2.0, max(0.5, (pace if pace > 0 else p["pace_ref"]) / p["pace_ref"])) ** 0.5)
    C = max(p["floor"], 1.0 - t / scale)
    if t > scale:
        C *= p["stuck_pen"]
    B = (0.0, p["b1"], p["b2"], p["b3"])[min(max(0, N - level), 3)]
    return (A * C + B * phi) * 1000.0


def policy_ours(g, phi):
    if not g.started:
        return ours_value(1, 0, 0.0, 0.0, g.N, phi) * OURS["fresh_mult"]
    return ours_value(g.level, g.act_on_level, g.tok_on_level, g.pace(), g.N, phi)


def policy_oracle(g, phi):
    """Clairvoyant upper bound: knows the tokens to its next clear from the trace."""
    if not g.started:
        return FRESH_BASE - g.queue_pos
    toks = 0.0
    for i in range(g.idx, len(g.steps)):
        toks += g.steps[i][0] * DIFFICULTY
        if g.steps[i][2]:
            gain = g.level * norm_N(g.N) + (0.5 if g.level < g.N else 0.0)
            return 1e6 * gain / max(1.0, toks)
    # no clear left in the trace: expected via hazard
    h = hazard_for(g.gid, g.level) / DIFFICULTY
    if h <= 0:
        return 1.0
    return 1e6 * (g.level * norm_N(g.N)) * h * 0.3


POLICIES = {"base": policy_base, "dprime": policy_dprime, "fair": policy_fair, "oracle": policy_oracle,
            "random": policy_random, "ours": policy_ours}
FADE = {"base": 0.2, "dprime": 0.4, "fair": 0.2, "random": 0.2, "ours": 0.4, "oracle": 0.2}


# ---- the event-driven run ---------------------------------------------------------------
def simulate(games, policy, slots=SLOTS, eta="sqrt", wall=WALL_BUDGET_S, cap=PER_GAME_CAP_S,
             allow_continue=True, fade=0.2):
    e = eta_curve(eta); ref = e(min(slots, SLOTS))
    tps = lambda active: TPS10 * (ref / e(max(1, min(active, 64))))
    t = 0.0
    heap = []            # (t_free, seq, game, tokens, clears, dt)
    seq = 0
    waiting = list(games)
    active = 0

    def phi_at(now):
        remaining = max(0.0, wall - now)
        window = fade * wall
        return min(1.0, remaining / window) if window > 0 else 1.0

    def eligible(g):
        return (not g.done) and g.slot_time < cap

    def start(g, now):
        nonlocal seq, active
        nxt = g.next_action(allow_continue)
        if nxt is None:
            g.done = True
            return False
        tokens, clears = nxt
        dt = tokens / tps(active + 1)
        active += 1; g.started = True; seq += 1
        heapq.heappush(heap, (now + dt, seq, g, tokens, clears, dt))
        return True

    def admit(now):
        nonlocal waiting
        waiting = [g for g in waiting if eligible(g)]
        while active < slots and waiting:
            phi = phi_at(now)
            best = max(waiting, key=lambda g: policy(g, phi))
            waiting.remove(best)
            if not start(best, now):
                continue

    admit(0.0)
    while heap:
        t_free, _s, g, tokens, clears, dt = heapq.heappop(heap)
        if t_free > wall:
            break
        t = t_free
        active -= 1
        trim = g.apply(tokens, clears, dt)
        if eligible(g) and not trim:
            start(g, t)
        elif eligible(g):
            waiting.append(g)
        if active < slots:
            admit(t)
    return games


# ---- populations --------------------------------------------------------------------------
def pool_by_game(traces, runs=None):
    pool = {}
    for rname, run in traces["runs"].items():
        if runs and rname not in runs:
            continue
        for gid, g in run.items():
            pool.setdefault(gid, []).append(g)
    return pool


def hidden_population(traces, n, seed, hard_frac):
    pool = pool_by_game(traces)
    rng = random.Random(seed)
    easy = [g for g in EASY if g in pool]; hard = [g for g in HARD if g in pool]
    games = []
    for i in range(n):
        gid = rng.choice(hard) if (hard and rng.random() < hard_frac) else rng.choice(easy)
        games.append(Game(f"{gid}#{i}", rng.choice(pool[gid]), queue_pos=i, seed=seed))
    return games


def run_population(traces, which):
    return [Game(gid, g, queue_pos=i) for i, (gid, g) in enumerate(traces["runs"][which].items())]


def mean_score(games):
    return sum(g.score() for g in games) / len(games)


def total_levels(games):
    return sum(g.levels_cleared() for g in games)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--compare", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=HIDDEN_GAMES)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--eta", default="sqrt")
    ap.add_argument("--wall", type=float, default=WALL_BUDGET_S)
    ap.add_argument("--no-continue", action="store_true")
    ap.add_argument("--hazard-scale", type=float, default=1.0)
    ap.add_argument("--hard-hazard-scale", type=float, default=1.0)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--params", default=None, help="JSON file with OURS params")
    args = ap.parse_args()
    traces = json.loads(TRACES.read_text())
    hp, hg = build_hazards(traces)
    global HAZ_SCALE, HAZ_HARD_SCALE, DIFFICULTY
    HAZ_SCALE = args.hazard_scale; HAZ_HARD_SCALE = args.hard_hazard_scale; DIFFICULTY = args.difficulty
    if args.params:
        OURS.clear(); OURS.update(json.load(open(args.params))["params"])
    print("population clear hazard (clears per 10k tokens) by level:",
          {k: round(v * 1e4, 2) for k, v in sorted(hp.items())})

    if args.validate:
        print("[validate] no contention, no continuation -> demo levels:")
        for which in ("copy-franzen-m2", "franzen-rep", "franzen-rep2", "hardtrace"):
            games = run_population(traces, which)
            simulate(games, policy_base, slots=999, wall=10 * WALL_BUDGET_S, allow_continue=False)
            print(f"  {which:>16}: sim {total_levels(games):3d}  trace {sum(g['levels'] for g in traces['runs'][which].values()):3d}")
        return

    if args.calib:
        run = traces["runs"]["hardtrace"]
        obs = (sum(g["levels"] for g in run.values()), sum(len(g["steps"]) for g in run.values()))
        print(f"[calib] real hardtrace run (25 games, 10 slots, 1500 s wall): levels {obs[0]} actions {obs[1]}")
        for pname in ("base", "dprime", "fair"):
            games = run_population(traces, "hardtrace")
            simulate(games, POLICIES[pname], slots=10, wall=1500.0, cap=1500.0, allow_continue=False, fade=FADE[pname])
            touched = sum(1 for g in games if g.started)
            print(f"  sim {pname:>6}: levels {total_levels(games):3d} actions {sum(g.idx for g in games):4d} games started {touched:2d}")
        return

    policies = args.compare or ["fair", "base", "dprime", "ours"]
    print(f"[compare] {args.n} games, hard_frac {args.hard_frac}, {args.trials} seeds, eta {args.eta}, "
          f"wall {args.wall / 3600:.1f} h, continuation {'off' if args.no_continue else 'on'}")
    rows = []
    for pname in policies:
        ms, lv, deep = [], [], []
        for seed in range(args.seed0, args.seed0 + args.trials):
            games = hidden_population(traces, args.n, seed, args.hard_frac)
            simulate(games, POLICIES[pname], eta=args.eta, wall=args.wall,
                     allow_continue=not args.no_continue, fade=FADE[pname])
            ms.append(mean_score(games)); lv.append(total_levels(games) / len(games))
            deep.append(sum(1 for g in games if g.levels_cleared() >= 4) / len(games))
        rows.append((pname, st.mean(ms), st.pstdev(ms), st.mean(lv), st.mean(deep)))
        print(f"  {pname:>7}: mean score {st.mean(ms):6.2f} +- {st.pstdev(ms):4.2f}   levels/game {st.mean(lv):4.2f}   games>=L4 {st.mean(deep):4.2f}")
    return rows


if __name__ == "__main__":
    main()
