# The scheduler is the lever — offline simulator result (2026-10-07)

## Why this is the finding
Draw #71 (base Franzen copy) scored **26.88**, mid-pack. The public board now holds **953
teams in the 20–34 "copy band" (mean 27.9, sd 2.7)** — everyone cloning Franzen's
Milestone-2 notebook clusters here. The top-5 line is **39.11** and rising (37.94 on 10/6;
projected 42–46 by 11/2). Nothing on the 25-minute demo moves a copy out of the band,
because the demo gives each game its own stream: no contention, so allocation cannot
matter. Every depth/serving arm we have measured reads null for exactly this reason.

**The field found the one lever that does move it.** On 10/4–10/6 Osaka Metropolitan
"Affectify" (shiiin9) replaced Franzen's slot-priority scheduler with a tuned formula
"D'" (`priority = A·M·C + B·φ`) — his harness, model, prompts all unchanged — and scored
**31.54 in a single submission** vs Franzen's 27–31 band. It is now copied across the
board (sujanmajhisuzan 31.93, amatlas, daydreamerthinks, lwq255, sigeward). The full D'
source is embedded verbatim in their public notebook (MIT; derivation withheld, formula
not). They found it with their own offline simulator, `arceval/sim/simulate.py` — not GPU
time.

The scheduler is the ONLY lever that (1) acts solely in the 110-game hidden run, so it
separates leaderboard rank from the demo; (2) is tunable **offline, on CPU, with no paid
compute** — it fits "maximise the Kaggle quota, no paid compute" exactly; (3) is genuine
original-algorithm territory (our own allocation policy, our own simulator).

## Our instrument: `scripts/sched_sim.py`
A discrete-event simulator of the 10-slot shared-server run. Each game is a real demo
trajectory (per-action generation tokens + level-clear marks, from our 120 trajectories in
`scratchpad/sched/traces.json`). A game holds a slot and generates until a context trim
(58k tokens), then yields; the gate fills free slots by a priority **policy**; score is the
real contest formula. Cost model: an action's wall = its generation tokens ÷ per-stream
throughput, calibrated to **61 tok/s at 10-way** (tight: p25–p75 58–64 over 120
trajectories). The recorded per-action `wallclock` is contention-inflated latency (~30×
the token work, mostly queueing — the known serving bottleneck) and is never summed.

**Validation:** with slots unbounded (no contention) the simulator reproduces each demo
run's levels **exactly** — 44 / 50 / 40 for copy-franzen-m2 / rep / rep2.

## Result: the lever's magnitude
Hidden-run model: 110 games resampled from the 25 demo games, a fraction replaced by
synthetic deep "slot-sink" games (the hard games the demo excludes and the hidden set
contains), 12 seeds, across the throughput band linear/sqrt/batch.

| policy | mean score | levels/game | vs fair |
|---|---|---|---|
| fair (round-robin) | 8.80 | 0.96 | — |
| random | 8.65 | 0.94 | −2% |
| Franzen base priority | 13.76 | 1.39 | **+56%** |
| D' (field, as benchmark) | 13.76 | 1.39 | +56% |
| marginal (ours, v0) | 12.93 | 1.26 | +47% |

(25% synthetic sinks; absolute scores are low because pure sinks contribute ~0 and are not
the real mix. The **relative** ranking is the robust output, identical across the eta band.)

**Reading.** Smart priority ≈ **+56% over naive round-robin** on a contested run — this is
why the copy pack exists (most copies ship Franzen's base priority) and why retuning it
breaks out. Our marginal-value policy v0 **underperforms** Franzen's base (it over-drops
deep games); it needs real tuning. Base and D' **tie here** because synthetic sinks are
trivially deprioritised; the fine separation that is worth D''s +4 on the board needs
realistic deep-but-slow games — which is the one thing the simulator lacks.

## The one missing ingredient → the GPU quota's best use
The simulator can only see the 25 easy demo games (all clear fully) plus synthetic sinks
(never clear). It cannot see the **spectrum** — games that clear 6–8 levels only if fed
enough, the tension a good policy actually resolves. Those are the **15 games the demo
excludes** (bp35, cd82, cn04, dc22, g50t, ka59, lf52, ls20, m0r0, s5i5, sk48, sp80, su15,
tn36, wa30). Collecting their real trajectories on the Franzen chassis is the highest-value
use of the ~30 GPU-h/week — **not** more null depth-prompt arms. ~2–3 demo runs with the
exclusion list cleared give the token/level traces the policy search needs.

## Honest ceiling
D' reached ~31.5 from a 27 base: ~+4. A better allocation might reach **~33–36** (top
~20–40 at today's lines). Top-5 is **39+ and rising to 42–46**. The teams above 37 (Tufa
55.9, Chen 48.6, Majkel 42.7, …) run their own models/harnesses, not Franzen copies; the
best Franzen-derivative on the board is the D' family at ~31.9. **Scheduler alone cannot
reach top-5 from a Franzen base** — that needs the scheduler PLUS a model/harness edge we
cannot train on Kaggle-only compute. The scheduler is nonetheless the correct leaderboard
play (it is the only move that leaves the band) and the centerpiece of the paper.
