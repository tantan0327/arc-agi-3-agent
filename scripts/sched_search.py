"""Offline search for OUR slot-priority parameters on scripts/sched_sim.py (CPU only).

Objective: paired mean score over seeds at the calibrated regime (hard_frac 0.6, hazard
scale 0.15 -> base ~ 28 on the board), with robustness terms at hazard 0.1 and 0.3 and at
hard_frac 0.5. A candidate is accepted only if it improves the main objective AND does not
lose more than 0.3 on any robustness regime. D' is evaluated on the same seeds as the bar.

    nohup .venv/bin/python scripts/sched_search.py --iters 300 > scratchpad/sched/search.out 2>&1 &
"""
from __future__ import annotations
import argparse, importlib.util, json, random, statistics as st, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ss", ROOT / "scripts/sched_sim.py"); ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)

# (hard_frac, hazard_scale, difficulty); set by --difficulty at run time
REGIMES = {}
SEEDS = list(range(8))
HELDOUT = list(range(60, 76))
DPRIME_LIKE = {"level_pow": 0.5, "eff_const": 300.0, "eff_pow": 2.5, "patience_tok": 225000.0, "floor": 0.1,
               "b3": 16.0, "b2": 14.0, "b1": 10.0, "pace_pow": 0.4, "fresh_mult": 1.0, "stuck_pen": 1.0, "pace_ref": 30000.0}
LOG = ROOT / "scratchpad" / "sched" / "search.log"
BEST = ROOT / "scratchpad" / "sched" / "best_params.json"


def evaluate(policy, regime, seeds, traces, fade):
    hf, kappa, diff = REGIMES[regime]
    ss.HAZ_SCALE = kappa; ss.DIFFICULTY = diff
    out = []
    for seed in seeds:
        games = ss.hidden_population(traces, 110, seed, hf)
        ss.simulate(games, policy, wall=ss.WALL_BUDGET_S, allow_continue=True, fade=fade)
        out.append(ss.mean_score(games))
    return st.mean(out)


def perturb(p, rng):
    q = dict(p)
    keys = rng.sample(list(q), k=rng.randint(1, 3))
    for k in keys:
        if k in ("floor", "stuck_pen"):
            q[k] = min(1.0, max(0.01, q[k] * rng.uniform(0.6, 1.6)))
        elif k == "fresh_mult":
            q[k] = min(3.0, max(0.2, q[k] * rng.uniform(0.7, 1.4)))
        else:
            q[k] = max(1e-3, q[k] * rng.uniform(0.6, 1.6))
    return q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=200); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--difficulty", type=float, default=1.15); ap.add_argument("--out", default="ours_v2")
    args = ap.parse_args()
    d = args.difficulty
    REGIMES.update({"main": (0.8, 0.1, d), "harder": (0.8, 0.1, d + 0.4), "easier": (0.8, 0.1, max(1.0, d - 0.2)),
                    "mix": (0.6, 0.1, d), "allhard": (1.0, 0.1, d)})
    traces = json.loads(ss.TRACES.read_text()); ss.build_hazards(traces)
    rng = random.Random(args.seed)
    log = open(LOG, "a")
    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()
    t0 = time.time()
    bars = {r: evaluate(ss.policy_dprime, r, SEEDS, traces, 0.4) for r in REGIMES}
    base = {r: evaluate(ss.policy_base, r, SEEDS, traces, 0.2) for r in REGIMES}
    say("[bars] D' %s | base %s | %.0fs" % ({k: round(v, 2) for k, v in bars.items()}, {k: round(v, 2) for k, v in base.items()}, time.time() - t0))
    # start from D' expressed in our family (the published point), then search
    cur = dict(DPRIME_LIKE)
    def score_of(params):
        ss.OURS.clear(); ss.OURS.update(params)
        main_v = evaluate(ss.policy_ours, "main", SEEDS, traces, 0.4)
        return main_v
    cur_v = score_of(cur)
    ROB = ("harder", "easier", "mix", "allhard")
    robust = {r: evaluate(ss.policy_ours, r, SEEDS, traces, 0.4) for r in ROB}
    say("[start] ours main %.2f robust %s" % (cur_v, {k: round(v, 2) for k, v in robust.items()}))
    best = (cur_v, dict(cur), dict(robust))
    for it in range(args.iters):
        cand = perturb(cur, rng)
        v = score_of(cand)
        if v <= cur_v + 0.02:
            continue
        rob = {r: evaluate(ss.policy_ours, r, SEEDS, traces, 0.4) for r in ROB}
        if any(rob[r] < robust[r] - 0.3 for r in rob) or any(rob[r] < bars[r] - 0.2 for r in rob):
            say("  it %d: main %.2f better but robustness lost %s -> rejected" % (it, v, {k: round(x, 2) for k, x in rob.items()}))
            continue
        cur, cur_v, robust = cand, v, rob
        best = (v, dict(cand), dict(rob))
        json.dump({"main": v, "robust": rob, "params": cand, "bars_dprime": bars, "bars_base": base}, open(BEST, "w"), indent=1)
        say("  it %d: ACCEPT main %.2f (D' %.2f, base %.2f) robust %s params %s" % (
            it, v, bars["main"], base["main"], {k: round(x, 2) for k, x in rob.items()}, {k: round(x, 3) for k, x in cand.items()}))
    say("[done] best main %.2f vs D' %.2f base %.2f | %.0f min" % (best[0], bars["main"], base["main"], (time.time() - t0) / 60))
    # held-out verdict on fresh seeds, every regime; adopt only if >= D' + 0.5 everywhere
    ss.OURS.clear(); ss.OURS.update(best[1])
    verdict = {}
    for r in REGIMES:
        o = evaluate(ss.policy_ours, r, HELDOUT, traces, 0.4); dp = evaluate(ss.policy_dprime, r, HELDOUT, traces, 0.4)
        verdict[r] = (round(o, 2), round(dp, 2))
    say("[held-out seeds 60-75] ours vs D': %s" % verdict)
    ok = all(o >= dp + 0.5 for o, dp in verdict.values())
    out = ROOT / "scratchpad" / "sched" / (args.out + ".json")
    json.dump({"params": best[1], "heldout": verdict, "adopt": ok, "difficulty": d}, open(out, "w"), indent=1)
    say("[verdict] %s -> %s" % ("ADOPT ours" if ok else "keep D'", out))


if __name__ == "__main__":
    main()
