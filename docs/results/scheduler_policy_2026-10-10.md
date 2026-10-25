# Our slot-priority policy beats D' and Franzen's base on held-out simulations (2026-10-10)

Instrument: `scripts/sched_sim.py` v2 — event-driven, cost = generation tokens ÷ 61 tok/s,
fresh-game band, refresh-at-pump, tail fade; hazard continuation after a trace ends.
Validation: no-contention replay reproduces every demo run exactly (44/50/40/25); the real
25-game/10-slot/25-min run is reproduced at 25 levels / 937 actions (observed 25 / 938).

Search: `scripts/sched_search.py`, 300 iterations of random local search over a 12-parameter
family (contains base and D' as points), seeds 0–5, with robustness vetoes. Patch:
`scripts/patches/priority_form.py` (CC0; `scripts/test_priority_form_stub.py` 6 checks pass
against the patched harness; formulas identical to the simulator's to 1e-15).

Held-out (seeds 20–35, never used in the search), mean contest score over 110 games:

| regime (hard_frac, hazard, hard-hazard) | fair | base | D' | ours v1 | ours − D' |
|---|---|---|---|---|---|
| 0.6, 0.15, 1.0 (calibrated) | 30.8 | 33.1 | 34.4 | **36.6** | +2.2 |
| 0.6, 0.15, **0.0** (hard games never clear) | 27.5 | 30.9 | 31.8 | **34.8** | +3.0 |
| 0.6, 0.15, 0.3 | 28.2 | 31.1 | 32.1 | **35.1** | +2.9 |
| 0.7, 0.15, 1.0 (harder set) | 25.4 | 27.9 | 28.9 | **30.4** | +1.5 |
| 0.5, 0.30, 1.0 (easier set) | 46.5 | 50.7 | 53.4 | **55.3** | +1.9 |
| 0.6, 0.08, 1.0 (almost nothing clears) | 26.4 | 27.6 | 28.1 | 28.3 | +0.2 |

D' − base in the simulator: +1.0 to +2.7 — the board says +2.4 (three public D' draws vs the
copy band), so the instrument's scale is about right. Ours − D' is +1.5 to +3 everywhere the
run is not hopeless, including the worst case for patience (hard games unclearable).

What the policy does (ours v1): `A = (1 + 1.9·(ℓ−1))·norm(N)·(841/(841+a))^1.19·pace^0.61`,
`C = max(0.053, 1 − t/469k)`, `B = 4.9/17.5/9.4` (by levels remaining) × tail fade 0.4,
fresh games priced at 1.71× the formula. In words: start every game once, then concentrate
slot time on the games that are already deep and still progressing, be very patient with
them, and ignore action-efficiency when deciding who plays. Honest caveats: the hidden
population is modeled from 25 public games; continuation after a 25-min trace is an
extrapolation (the 60-min hard-game traces, hardtrace2a/b, replace it tonight).


## Update, later on 2026-10-10: the 60-minute traces and the ceiling of allocation

The two 60-minute runs (18 "hard"/slow games, each with its own slot) cleared 69 levels:
cd82 6/6, cn04 6/6, sc25 6/6, ka59 6/7, re86 6/8, m0r0 5/6, dc22 4/6, ls20 4/7, sp80 4/6 …
Median tokens to the first clear 20k; gaps between later clears 30–80k tokens, up to 180k.
These games are not stuck — they are **slow**: Franzen's base patience (a 62k-token
Gaussian) abandons them mid-level; D' (225k linear) does not. That is where D''s +2.4 to
+3.9 on the board comes from.

With these traces folded in, the simulator over-predicted the board (base ≈ 36–41), so a
difficulty factor (token inflation of every action; hidden games are harder than public
ones — Franzen's public-25 mean was 46.5 vs 27.9 semi-private) was added and calibrated:
at hard_frac 0.8, difficulty 1.0–1.15 the simulator gives base 27.6–26.9 and D' 29.8–28.5
(board: 28.15 and ~30.5). A clairvoyant **oracle** policy (knows each game's tokens to its
next clear) reaches only **31.7–30.5** there: the whole allocation lever is worth about
**+4 over base and +2 over D'** when total slot time is the binding constraint. Our v1
(searched on the earlier extrapolation) collapses under difficulty (loses to base) — it
was overfitted to a constant-hazard world; v2 is being searched on the calibrated regime
with D' as the starting point and a held-out adoption rule (≥ D' + 0.5 everywhere).

Consequence for the top-5 goal: a perfect scheduler on a Franzen base lands ~32 on the
board. The remaining gap to 42+ must come from tokens-per-level efficiency of the agent
(e.g. the 31% of turns that take no action), a stronger model, or a public breakthrough.

**v2 verdict (night):** searched on the calibrated regime from the D' point; held-out (seeds
60–75) ours − D' = +0.4…+0.9 across main/harder/easier/mix/allhard, i.e. inside the +2
oracle headroom but not a reliable edge. Board gamble #2 therefore ships the clean-room D'.
