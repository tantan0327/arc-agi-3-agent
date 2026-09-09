# ARC-AGI-3: a symbolic agent, and what happened when it was measured against the field

Two things are documented here, and it matters which is which.

**Ours.** `agent/my_agent.py` is a symbolic agent written from scratch for this
competition: ordinary Python that looks at a grid, forms hypotheses about what the
buttons do, and acts on them. Nothing is trained and there is no model to download.
`duck_delta/` is a small set of perception layers, also ours, described below. All
of it is published under **MIT-0** (see `LICENSE`), as the rules require of code
authored by the submitter.

**Not ours.** The submission that produces our leaderboard score is built on the
Tufa Labs "duck harness", the Milestone 1 winning entry, written by Harold Bessis,
Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal Tesnar and Stefano Viel. It
is public on Kaggle and its solver ships in the dataset
`jeroencottaar/taaf-kaggle-source-share` under **MIT**. We did not write any of it.
The competition rules ask for third-party code under a licence permitting public
sharing, naming Apache-2.0 and GPLv3 as sufficient; MIT clears that. Our copy is
`reference/taaf-duck/upstream.ipynb`, and `scripts/build_duck_notebook.py` shows
every line we change.

## The state of the solution as submitted (updated 2026-09-09)

Three facts describe where this ended up, and all three are measurements.

**What produces our leaderboard score** is `tantan0327/arc3-flashnext-asis`, a
byte-faithful copy of a public kernel lineage (keithtyser's Duck + Qwen3.8
-Flash-Next NVFP4 MTP serving stack), redrawn on a schedule. We verified its
licence chain before adopting it: the runtime is the official `vllm/vllm-openai`
Docker image (Apache-2.0, confirmed by image digest), and the weights are
RadixArk's NVFP4 quantisation of Qwen3.8-Flash-Next under the Qwen Community
License 1.0, whose restrictions (>100M-MAU products, model-as-a-service) do not
touch competition use. Running an unmodified public artifact and saying so
plainly is the honest outcome of our measurement program, not a placeholder:
the leaderboard is a max-statistic over draws (§"What the competition itself
does"), and the field's best public recipe, drawn repeatedly, beats every
modification we tested but one.

**The one modification that beat its baselines** — of more than fourteen
submitted across the season — was an engine substitution: the same public
recipe served through a newer vLLM with multi-token prediction enabled (our
locally built 196-wheel Linux closure), which scored 2.27 on the board against
its own family's 1.9-2.2 band, and +7 levels offline. Faster, losslessly
identical decoding buys levels only when it changes what fits inside the
per-game clock; every other speed/memory/prompt intervention did not.

**Why we stopped modifying.** Eighteen measured eliminations — throughput,
retention, imitation fine-tuning twice over, prescribed reasoning protocols,
executable world models in two forms — converged on one result: on these games
98.3% of a solving trajectory's decisions occur at a board state never seen
before in that game, so what binds is the model's ability to reason about a
novel configuration, and no harness engineering raises that ceiling. The full
analysis, with reproduction scripts for every figure, ships with our paper;
the elimination story below (unchanged from earlier drafts) is how we got
there.

## Why the honest answer is a fork

The symbolic agent reached **0.25** on the public leaderboard after a week of work.
The duck harness, downloaded and pushed unmodified, scored **0.98, 1.14, 1.11 and
1.23** on four submissions of a byte-identical build. Four to five times our own
agent, for no work at all. (Resubmitting one kernel version re-runs it rather than
returning a cached score, which is what makes four samples of one build possible at
no GPU cost.)

Continuing to polish a 0.25 agent while a 1.06 one was sitting in public would have
been a decision about pride rather than about the problem. So the agent below is
kept, documented and published — it is real work and its measurements are real —
but the submissions are the fork, and our contribution is what we add to it.

That contribution is, so far, **not demonstrated to help**. Four modifications have
been submitted and **all four scored below every unmodified submission**:

```
unmodified   0.98  1.14  1.11  1.23      mean 1.115
modified     0.58  0.69  0.72  0.42      mean 0.602
```

The probability of that ordering by chance is 1/C(8,4) = **1.4%**. What finally
explained it is not a property of the changes but a property of the metric, and it
is written up under "Actions are the budget" below. The failures are documented at
the same length as everything else, because on this problem they were the only
thing that produced a usable model.

## The scoring formula is the whole problem

Most of what follows only makes sense once the metric is on the table, and the
metric is not "how many levels did you finish". From
[ARC's own methodology](https://docs.arcprize.org/methodology.md), matching
`arc_agi/scorecard.py` line for line:

```
level_score = (human_baseline_actions / ai_actions) ** 2      capped at 1.15
game_score  = weighted mean of level scores, weight = 1-indexed level number
final       = mean of game scores, 0-100%
```

The baseline is the upper median of first-time human players, by fewest
actions. Three consequences drive every design decision here:

1. **Actions are the score, squared.** Finishing a level in twice the human
   baseline scores 25 of 100; ten times, 1.0; a hundred times, 0.01.
2. **Later levels count for more**, and levels never reached still sit in the
   denominator, so depth is worth more than it looks.
3. **The final score is a mean over games**, so a game that never finishes a
   level contributes zero however efficiently it played.

Points 1 and 3 pull against each other, and getting that tension wrong is the
most expensive mistake recorded in this repository — see "What did not work".

## Architecture

Nine layers, each of which can be switched off without breaking the ones below.

| Layer | What it does |
|---|---|
| Perception | 4-connected flood fill into coloured objects; clicks are aimed at object centroids rather than uniformly at random |
| State key | SHA-1 over sorted `(colour, centroid)`, excluding colours learned to be noise — so the world model keys on situations, not pixels |
| Object tracker | Persistent ids across frames via three-pass matching (exact, moved-by-shape, recoloured-by-overlap) |
| World model | Per-`(state, action)` try/changed/scored counts, a discovered state graph, and BFS to the nearest scoring state |
| Effect model | Pooled "does clicking this colour do anything at all", so inert clicks are not re-tested on every new board |
| Motion model | Learns `action -> (dr, dc)` per tracked object, identifies the avatar as the object whose motion depends on which key was pressed, and flags counters/timers as noise |
| Contact model | Learns which colours kill and which pay, scored by **lift against the board-wide rate** rather than absolute frequency |
| Navigator | BFS in avatar-position space, learning walls from moves that did not land where the offset predicted |
| Policy | Lookahead (off) → navigate → untried → known → planned → random |

Two details are worth calling out because both were bugs first.

**Lift, not frequency.** The first contact model blamed the background colour
for every death, because the background is beside the avatar whenever anything
happens. Scoring on the ratio to the board-wide rate fixes this: a colour
present everywhere scores 1.0 and is ignored.

**Every observation counts, not just the interesting ones.** The same model
initially recorded surroundings only when something happened, so every
observation was a death, the base rate was 100%, and the lift test answered
1.0 for everything. Steps now retire from the window as ordinary observations.


## What did not work — the symbolic agent

More was learned from the rejections than the acceptances, and they are kept
in the git history with their measurements rather than quietly deleted.

**Forward simulation (three rejections).** The games ship their own source, so
a private mirror can be held frame-identical and candidate moves tried in
imagination. It costs a ~12 ms deepcopy per candidate. Rejected twice on level
counts; re-enabled when the real formula showed it turning 208 actions per
level into 40 (arc_score 0.213 → 0.619); then **submitted, and the leaderboard
went 0.25 → 0.07.** The formula was right and the inference was wrong: with 18
of 25 games never finishing a level, a marginal action buys the chance to
crack a game at all, not efficiency on one already solved, and cutting
throughput 3.4x spends exactly that budget. Efficiency is the lever *after*
most games score once.

**Intrinsic goals.** Treating "an object vanished next to the avatar" as a
stand-in for a level completion, to escape the chicken-and-egg where goal
learning needs a success that has never happened. Measured as a loss
(est_score 1.244 → 0.964, four games worse, none better). A navigation game is
finished by *arriving* somewhere, and nothing is consumed when you arrive.

**Dropping the avatar from the state key.** Movement games produce 71-86%
unique states, which leaves the world model inert. Removing the avatar's own
centroid — the one object guaranteed to move every step — did not reduce
uniqueness at all: the boards have many moving objects and the avatar is only
one of them.

## What we add to the fork

Four perception layers in `duck_delta/`, spliced into the harness's Python-tool
sandbox by `scripts/build_duck_notebook.py`. The sandbox cannot import anything
project-level — which is why upstream pastes its own segmentation module in as
source — so these paste the same way, and each splice asserts it matched before
the result is compiled and installed. A silent no-match would upload a notebook
that looks patched, runs as the unmodified harness, and returns a baseline score
a day later looking like the idea failed.

| Layer | What it tells the model | Measured |
|---|---|---|
| `motion.py` | What each directional action does: "UP moves colour 9 by (-5,0), 62 observations" | Recovers ls20's scheme exactly; a coherent scheme on 11 of 20 games, silent on the 10 click-only ones |
| `click_effect.py` | Which colours have never responded to a click | ft09's wasted clicks 86% → 15%; overall 14% → 7% |
| `correspondence.py` | Rare colours marking points on two objects that belong together | Names cn04's plug and socket contacts with no other marker on the board |
| analyzer temperature | — | 0.6 → 1.0: the two highest of eight offline runs; median completed-level cost 1.38x human → 0.97x |

The first three share a property that the failures did not: **they state a
measurement, not a reading of what the board means.** "ACTION1 moved this by
(-5,0), twelve times out of twelve" is either true of the run so far or it is not.
There is no theory in it for the model to chase.

## What did not work — the fork

**Raising the per-game wall clock.** The harness gives each game 7,920 s and, with
every game running concurrently, uses about 2.4 of the 9 hours the deployed bundle
assumes. Raising it to 25,200 s **scored 0.58 against 0.98**. The reasoning had been
that completed levels lock in, which the scorecard confirms — `add_level` is called
once per level of the game whether or not it was reached, so the weight denominator
is fixed. Nothing in the formula explains the loss and Phase B logs are not
retrievable, so it remains undiagnosed.

**Correspondence markers as a legend.** The prompt line carried two suggestions:
that marked points belong together, and that one group might be a legend whose
order is the answer. cn04 scored for the first time; sk48, the game the legend
clause was written for, went from 91 actions to 1,225 and still finished on zero.
Split apart and measured alone, the pairing half did not reproduce cn04's score
either. Rejected.

**Click-effect tally.** The mechanism works — ft09 stopped aiming at colours that
never respond, waste fell from 86% to 15% — and it did not convert into score.
ft09 saved 71 clicks and still finished on zero. Submitted, **0.69**.

**Swapping the analyzer model.** Qwen3.6 shipped exactly two open-weight
checkpoints: the 27B dense the harness uses and 35B-A3B, a sparse MoE with 3B
active parameters. Same generation, both Apache-2.0, and the MoE mirror is on
Kaggle so no upload was needed. It had the strongest local evidence any change has
had — at an identical 1,800 s Phase A budget, same GPU, only the model changed:

```
generated tokens/sec   207.37  ->  652.83     3.1x
actions over 25 games     313  ->   1258      4.0x
```

Submitted, **0.42** — the worst result of the campaign. Full write-up below;
briefly, throughput was the wrong objective and the run that measured it was
truncated in a way that inverted the ranking.

## Actions are the budget, not the resource

This is the model the four failures produced, and it is the most useful thing in
this repository.

`level_score = (human_actions / ai_actions)² × 100`. Taking twice the actions for
the same level scores 0.25x; four times scores 0.06x. **Coverage enters the score
linearly and efficiency enters it squared**, so a change that finishes more levels
can still lose, badly, if each one costs more actions. That is exactly what the
model swap did: it finished *more* levels than the baseline (4 against 3 in the
same Phase A) and earned a third as much, clearing them at 0.41x human efficiency
against the baseline's 1.00x.

Two of the four failures bought actions and lost — the 25,200 s clock and the
model swap. Neither the wall clock nor the token rate is the binding constraint;
the harness's own diagnostic reports `gave_up`, not a timeout.

**Why Phase A hid this.** Phase A's own mean score cannot rank builds, because at
1,800 s the run is truncated mid-play and a faster model simply gets further: the
MoE's Phase A mean was 0.49 against the baseline's 0.34, the exact opposite of the
board. Ranking by that number, or by tokens/sec, is ranking by progress per second
in a competition scored on progress per action.

`scripts/action_gate.py` is the fix. It rebuilds each finished level's score from
the recorded human baseline and sums them — the metric's own arithmetic, applied
only to levels the agent actually finished, so it does not care where the
truncation fell:

| build | earned | predicted | actual |
|---|---|---|---|
| baseline 27B | 244 | 1.115 *(fitted)* | 1.115 |
| Milestone-1 winning config | 125 | 0.57 | not submitted |
| MoE 35B-A3B | 92 | 0.42 | **0.42** |

One fitted parameter and one confirming point is not a validated model. What is
structural rather than fitted is the shape, and the gate would have rejected both
non-baseline builds before they cost a slot.

## The human solutions are downloadable, and they are the metric's numerator

The 2.3 MB bundle the winning kernel attached — `jeroencottaar/taaf-kaggle-source`,
not the 450 KB `-share` one everything else uses — contains
`re_arc/dsl/official_human_replays/`: **22 of the 25 public games, 6,512 human
actions, segmented by level**, from the official ARC-AGI-3 human study. Its README
states these replays reach WIN within `baseline_actions` on every level, so the
scorecard reports exactly 100.0, asserted in Tufa's CI.

`scripts/replay_humans.py` plays them back offline. **All 22 replay fully in sync,
every level completed, 168 level completions captured** — which makes them free
ground truth for any hypothesis about what a level is asking for.

What they show so far:

- **Click targets are board-conditioned, not positional.** Only **7%** (median) of a
  level's click positions appear in any earlier level of the same game. Within a
  level positions repeat — s5i5 hammers one spot 12 times — but across levels
  almost nothing carries over. Humans re-read the target off the board every level.
- **They click small, rare things.** Over 1,842 clicks: the clicked colour occupies
  a median **2.1%** of the board, and the blob clicked has a median of **12 cells**.
  Two games invert this (su15 80.5%, r11l 59.7%) because there the target is a
  *place* — the background — not a thing. An agent needs both modes.
- **The agent is not wasting actions.** 87-89% of its actions change the board, and
  on the levels it finishes it is already at ~1.00x human efficiency. Its problem is
  not accuracy of contact.

Two hypotheses were tested against this data and **failed**, and are recorded so
they are not proposed again: that humans form a routine in early levels and reuse
it (later/earlier action ratio, median 1.83 — they spend *more* later), and that a
game's goal is stable enough across levels to extract from the first completion
(14 of 22 games share a colour-count invariant, 15 of 22 a colour-adjacency one;
too thin to build on).

## The measurement problem, which is most of the difficulty

Byte-identical builds of the harness score **0.98 and 1.14** on the leaderboard and
**1.27 and 1.38** offline. Across five identical offline runs, 5 of 25 games scored
every time, 5 never did, and **15 flipped**. The set of games that score is not
stable, so "the games that fail" is largely an artefact of whichever run you looked
at — a trap this project fell into and designed two deltas against.

**Offline runs do not predict the leaderboard.** Across every configuration with
both numbers:

| config | offline mean | leaderboard |
|---|---|---|
| upstream | 1.26, 1.38 | 0.98, 1.14, 1.11, 1.23 |
| click-effect tally | 1.27 | 0.69 |
| temperature 1.0 | 2.71, 1.51 | 0.72 |
| per-game clock 25,200 s | — | 0.58 |
| MoE 35B-A3B | 0.49 *(truncated)* | 0.42 |

Pearson r is **-0.39**. The configuration with the two best offline runs of eight
came back nearly the worst when scored. Six offline diagnostics and about fifteen
GPU hours decided nothing, and the loop they were feeding — measure locally, then
submit the winner — was measuring something the board does not reward.

For a long time the only visible pattern was that **every modified submission
scored 0.58-0.72 and every unmodified one 0.98-1.23**, with no overlap, and the
modifications had nothing in common except that each executes code on the scored
path where an unmodified build executes none. That reading suggested a null
control as the next experiment. It is no longer the leading explanation:
**action inflation accounts for three of the four losses on its own**, and it does
so quantitatively rather than by exclusion. The control is still worth a slot
eventually; it is no longer urgent.

The lesson that survives is narrower and more useful than "offline does not
transfer". Offline *scores* do not transfer. Offline *mechanisms* do — the
click-effect delta's waste rate moved 86% → 15% and that was real. The mistake was
choosing which mechanism to measure: throughput, when the metric charges for
actions.

### What the competition itself does

Several of these cost a submission, or nearly did.

- **Competition mode gives one run per environment.** `arc_agi/api.py` creates an
  environment only `if not scorecard.has_environment(env.game_id)`. This kills
  `bm.n_passes > 1`, which was queued to submit on the strength of
  `EnvironmentScoreList.score` returning `max(run.score for run in self.runs)`. That
  max is real; the path that would put two runs in the list is closed. A data
  structure supporting something is not a route to it.
- **The scored set is ~110 games and harder than the 25 public ones**, per the
  competition forum. That is the likeliest reason local diagnostics do not transfer.
- **Errored submissions do not spend the daily allowance** — seven errors and one
  success on one day, blocked only after the success.
- **A kernel iterated over many versions can start failing every scored rerun**
  regardless of its code, while the identical code on a fresh slug runs first time.
  `notebooks-duck-b/` exists for that.
- **0.25 on the board is roughly the random agent**, and it is also the median over
  2,162 teams. The symbolic agent above peaked at exactly 0.25.
- **A scored submission yields one number.** Phase B logs and outputs are not
  retrievable, and the API serves no kernel version but the latest.

What follows for anyone extending this:

- **A single submission cannot evaluate a change.** With one sample per arm and a
  spread of that size, detecting a 10% improvement needs tens of submissions, and
  there are about fifty in the whole competition.
- **Score is the wrong local indicator.** Prefer something aggregated over
  thousands of events: wasted-click rate moved 86% → 15% unambiguously in one run,
  where the mean score of the same run said nothing.
- **Repeat the baseline before believing any comparison.** Every wrong conclusion
  recorded here came from comparing one run against one run.
- **`caffeinate` long measurements on a laptop.** Elapsed time counts sleep. One
  measurement here read 3 h 37 m elapsed against 14 m 49 s of CPU, and a throughput
  "finding" built on that — a 77% decay, once treated as the largest available
  lever — was entirely an artefact.

## The harness is the other half

Three faults that cost real leaderboard points were invisible to local
testing, and all three lived in the gap between the test harness and the
competition's own orchestration:

- Agents run one thread per game against an HTTP gateway. An environment
  returning `None` raised, and because the threads are daemons, that ended the
  game silently — 16 of 25 died this way in the first realistic benchmark.
- `Swarm` builds all environments *sequentially* before starting threads. A
  harness that called `make()` inside each worker raced 25 startup resets and
  killed one game per run, which then accounted for nearly every dropped frame
  and forced reset in the health report.
- `Agent.main` reads `observation_space` each iteration, and on a jammed game
  that read returns a *stale but non-None* frame. A failure counter cleared by
  any non-None frame therefore never fired in production, making an escape
  mechanism dead code exactly where it mattered.

`scripts/evaluate.py` now mirrors `Agent.main` line for line and reports
health counters beside the score. `scripts/spend.py` reports where actions go
by decision mode, by level, and by whether the attempt ended in death.

## Reproducing

```bash
make setup                      # vendored framework + game sources
make evaluate                   # arc_score across the public games
.venv/bin/python scripts/spend.py     --budget 1000    # where actions go
.venv/bin/python scripts/diagnose.py  --budget 1000    # what each layer learned
make submit                     # preflight, build notebook, push to Kaggle

# the fork, and one submission cycle
.venv/bin/python scripts/build_duck_notebook.py   # deltas are gated at the top
make cycle-push TARGET=duck                       # push + wait for Phase A
make cycle-submit NOTE="..." YES=1                # spends the day's one slot
make cycle-watch                                  # poll for the score, record it

# before spending a slot on anything: does it earn more than the baseline?
make human-replays                                # the recorded human solutions
kaggle kernels output <kernel> -p out/            # the Phase A artifacts
make gate DIR=out GATE=244                        # non-zero exit means do not submit

# the human solutions themselves, replayed offline against the real engine
.venv/bin/python scripts/replay_humans.py         # 22/22 in sync, 168 completions
```

Three cautions for anyone extending this:

**Rank builds by `make gate`, never by Phase A's own mean score or by tokens per
second.** Phase A is truncated, so both of those reward getting further per second
in a competition that charges per action. Following them cost the two worst
submissions in this repository, 0.58 and 0.42.

**Check CPU time against elapsed before believing a duration.** An earlier
version of this file warned that throughput collapses above ~18 concurrent
games. That number, and a companion claim that throughput decayed 77% within an
hour, were both artefacts of the laptop sleeping mid-run — `ps` elapsed time
counts sleep. Wrap long measurements in `caffeinate` and compare `etime` to
`time`.

**A change that alters actions per second cannot be judged from a short local
run.** Five minutes of wall-clock is roughly 1,300 actions per game against
about 20,000 in the real rerun, and the two regimes disagree — which is
exactly how the simulator came to be shipped and reverted.
