"""Score a Phase A run with the competition's own arithmetic, before spending a slot.

Four modifications have been submitted and all four scored below every unmodified
submission (0.58 / 0.69 / 0.72 / 0.42 against 0.98 / 1.14 / 1.11 / 1.23). Each was
chosen on a local number that turned out to measure the wrong thing -- offline
mean score, wasted-click rate, tokens per second. This computes the number the
board computes.

**Read `benchmark.json`, not the Kaggle log.** The per-game table printed in the
log is a periodic snapshot written mid-run; it is stamped "ended: in progress" and
under-reports by one to two orders of magnitude. In the 9,600 s baseline run it
showed sk48 at 10 actions and 5,937 tokens, where `benchmark.json` has the real
figures: **1,192 actions and 58,645 tokens**.

`benchmark.json` also carries `base_actions_per_level` -- **the human baselines
the score actually divides by**. These are not the fastest-replay counts in
`official_human_replays`: sk48 level 1 is 61 against the replay's 14, because the
replays are the fastest human runs while the scored baseline is the upper median
of first-time players. An earlier version of this file used the replay counts and
was far too harsh.

So there is nothing to fit. Reimplementing `arc_agi/scorecard.py` over
`benchmark.json` reproduces the harness's own per-game scores **exactly, all 25 of
them, and the mean to three decimals** -- see `--verify`.

    level_score = min((baseline_actions / actions_taken)^2 * 100, 115)   if cleared
    game_score  = sum(level_score * (i+1)) / sum(1..L)                   over ALL levels
    final       = mean over games

Two consequences drive everything:

  * **Levels never reached stay in the denominator.** For a six-level game the
    weights total 21, so clearing level 1 alone caps at 115/21 = 5.5 while
    clearing levels 1 and 2 reaches (115 + 230)/21 = 16.4. **One more level on a
    game already solved is worth about three times a whole new game at level 1.**
  * **Coverage is linear, efficiency is squared** -- which is how the model swap
    finished more levels than the baseline and still scored a third as much.

Usage:
    python scripts/action_gate.py <dir-with-benchmark.json> [...] [--gate 0.916]
    python scripts/action_gate.py <dir> --verify      # check against the harness
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# The 9,600 s baseline diagnostic, the only run where the solver's own 7,920 s
# per-game cap binds rather than the outer budget. Its mean of 0.916 sits just
# under the 0.98-1.23 the same build scores on the board.
BASELINE_MEAN = 0.916
LEVEL_SCORE_CAP = 115.0


def game_score(run: dict) -> float:
    """`arc_agi.scorecard.EnvironmentScoreCalculator`, over one game's record."""
    base = run["base_actions_per_level"]
    taken = run["actions_per_level"]
    cleared = int(run["levels_completed"])
    weights = sum(range(1, len(base) + 1))
    if not weights:
        return 0.0
    earned = 0.0
    reachable = 0
    for i in range(len(base)):
        weight = i + 1
        # Only levels actually finished score; the rest still weigh the denominator.
        if i < cleared and i < len(taken) and taken[i] > 0:
            earned += min((base[i] / taken[i]) ** 2 * 100, LEVEL_SCORE_CAP) * weight
            reachable += weight
    return min(earned / weights, reachable / weights * 100)


def report(directory: Path, verify: bool) -> float:
    path = Path(directory) / "benchmark.json"
    if not path.exists():
        raise SystemExit(f"no benchmark.json under {directory} — this wants the "
                         f"directory `kaggle kernels output` wrote")
    runs = json.loads(path.read_text())["game_runs"]

    print(f"\n=== {directory}")
    print(f"{'game':<6}{'lv':>3}{'done':>5}{'score':>7}{'L1 acts':>9}{'human':>7}"
          f"{'ratio':>7}{'wall_s':>8}")
    scores = []
    for run in sorted(runs, key=lambda r: -game_score(r)):
        s = game_score(run)
        scores.append(s)
        base = run["base_actions_per_level"]
        taken = run["actions_per_level"]
        a1 = taken[0] if taken else 0
        b1 = base[0] if base else 0
        print(f"{run['game_id'][:4]:<6}{run['number_of_levels']:>3}"
              f"{run['levels_completed']:>5}{s:>7.2f}{a1:>9}{b1:>7}"
              f"{(a1 / b1 if b1 else 0):>7.1f}"
              f"{float(run['final_wallclock_seconds']):>8.0f}")

    mean = statistics.mean(scores) if scores else 0.0
    cleared = sum(1 for r in runs if int(r["levels_completed"]) > 0)
    deeper = sum(1 for r in runs if int(r["levels_completed"]) > 1)
    print(f"\n  games clearing level 1 : {cleared}/{len(runs)}")
    print(f"  games clearing level 2 : {deeper}/{len(runs)}   "
          f"<- worth ~3x a new game at level 1")
    print(f"  MEAN GAME SCORE        : {mean:.3f}   (baseline {BASELINE_MEAN})")

    if verify:
        # Per game, in the file's own order -- comparing two sorted lists would
        # hide a permutation, which is exactly the kind of agreement that is not
        # agreement.
        worst = max(abs(game_score(r) - float(r["final_score"])) for r in runs)
        print(f"  verify: largest disagreement with the harness's own per-game "
              f"score = {worst:.4f}")
    return mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--gate", type=float, default=None,
                        help=f"fail if the mean game score falls below this "
                             f"(the baseline scores {BASELINE_MEAN})")
    parser.add_argument("--verify", action="store_true",
                        help="check this reimplementation against the harness's "
                             "own per-game scores")
    args = parser.parse_args()

    results = [(d, report(d, args.verify)) for d in args.directories]

    if args.gate is not None:
        worse = [f"{d} ({m:.3f})" for d, m in results if m < args.gate]
        print()
        if worse:
            raise SystemExit(
                f"GATE FAILED: {', '.join(worse)} score below {args.gate:.3f}. "
                f"Every modification submitted so far has lost, and each was "
                f"chosen on a local number that was not this one. Do not spend a "
                f"slot on it.")
        print(f"gate passed: every run scores at least {args.gate:.3f}")


if __name__ == "__main__":
    main()
