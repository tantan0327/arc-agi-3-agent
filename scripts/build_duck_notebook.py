"""Build the Tufa Labs "duck harness" baseline kernel from the vendored upstream.

Why this exists: every measurement so far compares our agent against its own past
selves. The leaderboard says the top pack sits at 1.46-1.86 and we sit at 0.25, but
nothing local tells us what that gap is *made of*. Milestone 1's winning harness is
public, so it can be run under our own account and the answer read off directly —
one submission for a number that no amount of local benchmarking can produce.

Provenance and licence. The upstream notebook is `reference/taaf-duck/upstream.ipynb`,
pulled from Kaggle kernel `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`.
It is a launcher only: the solver lives in the attached dataset
`jeroencottaar/taaf-kaggle-source-share`, which Kaggle records as **MIT**, authored by
the Tufa Labs team (Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit,
Michal Tesnar, Stefano Viel). ARC Prize's rules require our own code under CC0/MIT-0
and third-party code under any share-permitting licence, so MIT upstream is fine — but
MIT also requires the attribution to travel with the copy, which is what the notice
cell below is for. Do not delete it.

Caveat the upstream authors state themselves: this readable notebook is *not* the one
that scored 1.21 — that was version 21 of `jeroencottaar/taaf-duck-harness-kaggle`,
which the Kaggle API refuses to serve (403 on any non-latest version of someone
else's kernel). They attribute the difference to luck. So a score below 1.21 here does
not by itself refute the baseline.

Three changes are applied, all inside the upstream cell reserved for exactly this:

  1. An attribution cell, prepended.
  2. A Phase A budget cap. Unlike our own submission notebook, this one *plays* during
     an ordinary "Save & Run All" — offline against the bundled environment files, for
     `target.max_runtime_s` = 32,400 s. Nine hours of RTX Pro 6000 to prove the thing
     boots is not a good trade, so the customization hook shortens that. The cap is
     inside `if not TRUE_SUBMISSION`, so the competition rerun is untouched.
  3. The correspondence-marker delta: `duck_delta/correspondence.py` pasted into the
     Python-tool sandbox, hooked onto the line that builds a frame's segmentation, and
     one line added to the prompt that documents the runtime state. This is the only
     change that affects a scored run.

    .venv/bin/python scripts/build_duck_notebook.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "reference" / "taaf-duck" / "upstream.ipynb"
OUT = ROOT / "notebooks-duck" / "duck.ipynb"
# A standby kernel carrying the identical notebook. A kernel iterated over many
# versions has been reported to fail every scored rerun with a generic system
# error, independent of its code, while the same code pushed to a fresh slug ran
# first time. This one is here so that swapping is a one-word change rather than
# an afternoon spent under a daily deadline.
OUT_B = ROOT / "notebooks-duck-b" / "duck.ipynb"

# How long Phase A plays. It has no effect on a scored run — the branch below is
# guarded on TRUE_SUBMISSION — so this is purely how much GPU quota a push costs.
#
# 1,800 s proves the stack boots: wheelhouse installs, vLLM comes up, the 36 GB
# model loads, the solver plays. Raise it to 9,600 to turn Phase A into the
# per-game diagnostic instead: the solver's own `max_runtime_s_per_game` is
# untouched outside a submission, so an outer budget above it lets that cap bind
# and reproduces the scored configuration offline, minus the gateway. That costs
# about 2.4 h of RTX Pro 6000 and yields the per-game score/level/action/token
# table plus an mp4 of every game. Worth it when a change needs measuring; not
# worth it for a push that only restores a known configuration.
# Two passes need room for both: ~530 s of setup plus 2 x 7,920 s if they
# run in sequence. soft_end lands at budget - 600.
#
# Overridable from the environment so a diagnostic run does not need a source
# edit that could then be pushed by accident:
#     PHASE_A_BUDGET_S=9600 python scripts/build_duck_notebook.py
# The committed default stays at the cheap boot test.
PHASE_A_BUDGET_S = float(os.environ.get("PHASE_A_BUDGET_S", 1800.0))

# There is deliberately no submission-path change any more.
#
# Raising the solver's per-game wall clock from the deployed 7,920 s to 25,200 s
# scored **0.58 against the unmodified harness's 0.98**. The reasoning behind it
# was that completed levels lock in — which the scorecard confirms, since
# `EnvironmentScoreCalculator.add_level` is called once per level of the game
# whether or not it was reached, so the weight denominator is fixed and a level
# finished early keeps its score. That much was right. It still lost 41%, and
# nothing in the formula explains it, so the cause is operational: truncation,
# memory over a run three times longer, or session expiry. Phase B logs are not
# retrievable, so it cannot be diagnosed from here.
#
# The rule that follows, and the more expensive lesson: a submission needs either
# a measured local difference or an existing score behind it. Reading the code and
# reasoning forward is a hypothesis, not evidence. It has now cost two slots.

# What the model is told about the new field. It will not look for what it has not
# been told about, and the harness's own prompt is where every other runtime field
# is documented.
#
# This line used to carry a second suggestion — that one group of marks might be a
# legend naming the other, and that the legend's order was probably the order to
# work in. The two halves behaved completely differently. cn04, whose plug and
# socket carry the same pale blue on their contacts, went 0.00 -> 0.38 and scored
# for the first time. sk48, which is the game the legend clause was written for,
# went from 91 actions to 1,225 and still finished on zero: it chased the idea for
# the whole run.
#
# The difference is what each asks for. "Bring these two points together" names an
# action against something already on the board. "That group is a legend and its
# order is the answer" is a theory about what the game means, and a wrong theory
# survives contact with the board indefinitely — which is exactly the failure the
# transcripts show on these games anyway, a wrong world model held to the end.
# Adding another one was the wrong direction.
PROMPT_LINE = (
    "- `current_frame.segmentation['markers']` lists rare colours that look like "
    "correspondence markers: a colour used sparingly, in several separate places, "
    "sitting on more than one object. Each entry has `colour`, `cells`, `uniform` "
    "(all its blobs the same size, which is the strong case), `hosts` (the colours "
    "of the objects it sits on), `points` (each blob's `centroid`, `cells` and the "
    "colours it touches), and `reading`, a one-line summary. Treat a uniform marker "
    "as a strong hint that those marked points are meant to be brought into contact "
    "with each other.\n"
)

NOTICE = """\
## Provenance

This notebook is a copy of [Tufa Labs' duck harness]\
(https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner),
the ARC-AGI-3 Milestone 1 winning submission, run unmodified as a baseline measurement.

The solver is **not** ours. It was written by the Tufa Labs team — Harold Bessis,
Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal Tesnar, Stefano Viel — and ships
in the attached dataset `jeroencottaar/taaf-kaggle-source-share`, licensed **MIT**.

Changes from upstream, both in the single cell the harness reserves for them: a run
budget cap that applies only outside a competition rerun, and a perception addition —
rare colours that mark corresponding points on two different objects are computed
alongside the segmentation and reported to the model.
"""

# The analyzer's sampling temperature. The bundle's setup commands set 0.6.
#
# Worth testing because the dominant problem is not capability but consistency:
# across five identical offline runs, 5 of 25 games scored every time, 5 never
# did, and **15 flipped**. That instability is most of the spread between runs
# (offline 1.27 vs 1.38, leaderboard 0.98 vs 1.14 on byte-identical builds), and
# it is also where the headroom is — if the fifteen scored reliably the mean
# would rise by roughly 0.6, which is the difference between where we are and a
# prize position.
#
# The direction is genuinely unknown. Less temperature means less exploration,
# and the failure the transcripts show is a wrong idea held to the end, which
# less exploration could make worse. That is why it is measured rather than
# assumed. None means leave the bundle's value alone.
# Left at the bundle's value while the control-scheme delta is measured on its
# own. 1.0 is a separate finding and is submitted from version 11; mixing them
# would make the diagnostic unattributable.
ANALYZER_TEMPERATURE = None

# A control that changes nothing but runs during a scored submission.
#
# **Five** modifications have now been submitted and every one landed in
# 0.42-0.74, while the unmodified harness landed 0.98 / 1.11 / 1.14 / 1.23. The
# ranges do not overlap; an exact permutation test on the nine scores gives
# p = 0.0079.
#
# **Four of the five** share not their content -- a wall clock, a click tally, a
# control scheme, a sampling temperature, an early stop have nothing in common --
# but that each executes code in this hook *during the scored run*, and they sit
# in a tight 0.58-0.74. On an unmodified build the cell executes zero lines when
# TRUE_SUBMISSION is set. The fifth, the MoE swap (0.42), is not one of them: it
# rewrites setup_commands.json and its pushed notebook carries `# no delta` on the
# scored path, so it is a model-quality result and this control does not speak to
# it. (Checked against the pushed kernel, not git -- the flags below are always
# restored to off before committing, so the committed source never records what an
# arm actually contained.)
#
# The fifth, the early stop, is what forces this control to finally be run. It is
# the only modification that can be *proved* near-neutral and it lost anyway:
# replaying its own stopping rule over the unmodified full-length diagnostic cuts
# **zero** scoring games (every game that cleared level 1 did so within 150
# actions, against a 200 threshold), that diagnostic really is submission-length
# (median final_wallclock_seconds 7,921 s), its wrapper costs a cached-property
# read per call, and it lost no throughput (tokens/sec ratio 1.04). Offline said
# null, mechanism says null, throughput says null; the board said -0.375.
#
# This is that control: it imports a harness module and prints a length, on the
# scored path, and changes nothing. It is deliberately shaped like the other five
# -- import plus print -- so that a null result clears the modification *practice*
# and not merely "one arithmetic statement is free", which nobody doubted.
#
#   0.95-1.2  the hook is innocent, and the five changes each genuinely hurt
#   0.6-0.8   the hook is the cause, and all five measurements were confounded;
#             bisect import vs print next, and it is worth +0.4 on every arm
#
# Built and pushed as `arc3-duck-baseline-b` **v4** on 2026-08-13, so the flag is
# back off here: the arm lives on Kaggle, and leaving this True would let a stray
# `make duck` ship a control to the submission kernel.
DELTA_NULL_CONTROL = False

# A1: read out, for free, which half of the null control is guilty. Phase A only
# -- the body gates itself on TRUE_SUBMISSION, because printing is one of the two
# suspects and a probe that pays the tax would be measuring itself.
DELTA_HOOK_PROBE = False

# Phase A load test: how many game runs the solver keeps resident at once.
UNSEEN_GAMES = bool(os.environ.get("UNSEEN_GAMES"))
UNSEEN_CAP_S = float(os.environ.get("UNSEEN_CAP_S", "4500"))
DELTA_CONCURRENCY = int(os.environ["DELTA_CONCURRENCY"]) if os.environ.get("DELTA_CONCURRENCY") else None


# How many times each game is played in one run.
#
# The scorecard combines repeats of the same environment with **max**, not mean:
#
#     class EnvironmentScoreList:
#         @property
#         def score(self) -> float:
#             """Return the average score of the runs."""   # docstring says average
#             return max(run.score for run in self.runs)    # code takes the maximum
#
# and the upstream notebook sets `bm.n_passes = 1`. Given that 15 of 25 games flip
# between scoring and not across identical runs, a second pass converts a coin-flip
# into a best-of-two on exactly the games that are costing the most.
#
# The reason to measure rather than assume is time. `concurrency` is 28 against 25
# games, so two passes means 50 game-runs. Run together they each get half the
# tokens, which could easily cost more than the max recovers; run in sequence the
# whole thing takes about 4.4 h, inside the nine the bundle assumes. One run
# distinguishes those, and the per-game numbers say whether it was worth it.
#
# None leaves the upstream value alone.
# Cancelled before it was ever submitted. Competition mode gives one run per
# environment: arc_agi/api.py creates an environment only `if not
# scorecard.has_environment(env.game_id)`, so a second pass over the same game
# has nothing to attach to. The max(run.score for run in self.runs) that
# motivated this is real, but the path that would put two runs in that list is
# closed in the mode we are scored under -- a data structure supporting
# something is not the same as a route to it.
# Play each game this many times. The scorecard keeps, per game, the run with the
# most levels completed (arc_agi/scorecard.py, the best_idx loop), so extra passes
# are a best-of-N on exactly the games that flip -- and 5 of 25 flip between two
# runs of near-identical builds.
#
# It costs wall clock and not per-game time. The 50 runs of a two-pass build go
# into one list capped by `concurrency` (28), so they play in about two waves, and
# `_HarnessGameSession` -- which is where `started_at` is set -- is constructed
# inside `_play_one`, itself inside the semaphore. A game queued for a slot is
# therefore not burning its 7,920 s while it waits. Two passes come to roughly
# 16,400 s against the deployed `max_runtime_s` of 32,400 s.
#
# That resolves the concern this was shelved on in August, which assumed all 50
# runs shared one wall clock and so each got half the tokens. They do not.
N_PASSES = int(os.environ["N_PASSES"]) if os.environ.get("N_PASSES") else None
# Per-game wall clock, applied in the run cell alongside n_passes. The wave
# arithmetic that killed n_passes=2 at 7,920 s: ~110 games x2 / 28 = 8 waves;
# 8 x 3,500 = 28,000 s fits the 32,400 budget with margin. The 4,500 s accident
# (v2 unseen) measured the cost of a shorter cap at -2 levels of 20.
SOLVER_CAP_S = float(os.environ["SOLVER_CAP_S"]) if os.environ.get("SOLVER_CAP_S") else None
# S1 depth-over-breadth: play only these game prefixes, each with the whole
# clock. The mean over games is indifferent to breadth, levels were measured
# superlinear in per-game time up to 7,920 s, and past that nobody has looked.
DEPTH_GAMES = os.environ.get("DEPTH_GAMES") or None

# Whether the measured control scheme is spliced in.
#
# The harness leaves "what does ACTION1 do" to the model, re-derived from rendered
# grids every turn. Replaying the duck's own recorded play through
# duck_delta/motion.py recovers ls20's scheme exactly -- UP (-5,0), DOWN (5,0),
# LEFT (0,-5), RIGHT (0,5), 25 to 62 observations each -- and finds a coherent
# scheme on 11 of the 20 games with recorded play while staying silent on the 10
# click-only ones. dc22, zero in all five offline runs, comes back clean.
#
# Unlike the correspondence delta this states a measurement, not a reading of what
# the board means, so there is no theory for the model to chase.
DELTA_MOTION = False

# Whether the correspondence delta is spliced in at all. Off builds the harness as
# upstream ships it, which is what a repeat of the baseline needs.
#
# The first pair of runs gave mean score 1.30 without the delta and 0.95 with it,
# and that comparison cannot be read: the solver drives an LLM at temperature 0.6,
# so two single runs of 25 games separate nothing. Per game the swings ran both
# ways and were large — cn04 0.00 -> 0.38 and r11l 0.00 -> 4.76 against ft09
# 4.76 -> 0.00 and lp85 8.33 -> 2.78 — which is exactly the shape of a noisy
# measurement, whatever else is true.
#
# The obvious fix, fewer games with `bm.n_passes` raised, is not one: every game
# runs concurrently against a single GPU, so cutting 25 games to 5 hands each
# survivor several times the tokens and moves the regime being measured. Repeating
# the identical configuration is the only comparison that stays in place.
DELTA_CORRESPONDENCE = False

# Whether the click-effect tally is spliced in. This one is measured on recorded
# play rather than argued for: replaying ft09's own 98 clicks through it flags
# colours 0, 2, 4, 5 and 12 as never having responded, which accounts for 61 of
# that game's 84 wasted clicks. On cd82 and s5i5, where 84-100% of clicks work,
# it flags nothing — it suppresses only what has already been shown inert.
DELTA_CLICK_EFFECT = False

# ------------------------------------------------------ stop the hopeless games
#
# The first change that can only fire where the score is already zero.
#
# From the 9,600 s diagnostic: 25 games share one vLLM server for the 7,920 s
# per-game clock, so each gets ~68,000 generated tokens and a game's action count
# is just 68,000 / tokens-per-action. **Tokens are the currency.** Fifteen games
# spend theirs without ever clearing level 1 -- sk48 takes 1,192 actions against a
# baseline of 61, vc33 464 against 7 -- and because the server is shared, a game
# that stops early hands the rest of its throughput to the games still playing.
#
# Be honest about the size of that. Stopping at action N does not recover what was
# already spent, only the tail:
#
#     N     clearing games lost      tokens freed    per survivor
#     150   ar25 (0.09), re86 (0.02)     298,858     +37,357  (+55%)
#     200   re86 (0.02)                  191,023     +21,225  (+31%)
#     350   none                          44,237      +4,424   (+7%)
#
# 200 is the balance: it costs re86's 0.02 and frees a third more budget for the
# nine survivors. And the upside is lumpy in our favour -- level weights are
# 1..L, so **one game reaching level 2 is worth about +0.4 on the final score**,
# where the whole gap from 0.916 to a podium is ~0.7.
#
# The honest caveat, which the gate is there to settle: D3 found the ten games
# that clear level 1 have roughly the human's action budget for level 2 already
# (51 against 42) and fail anyway, so their problem may be comprehension rather
# than budget, in which case the freed tokens buy nothing. Two further reasons the
# estimate is optimistic: vLLM's aggregate throughput falls as batching drops, so
# a freed slot does not hand over its full share; and a uniform cap cannot tell a
# game that is stuck from one that is slow.
DELTA_EARLY_STOP = None      # set to an action count (e.g. 200) to enable


def early_stop_body() -> str:
    return f'''
_STOP_AFTER = {DELTA_EARLY_STOP!r}

# `should_stop` already exists on the per-game run object and is consulted in the
# play loop; extend it rather than adding a second stopping path. The guard is
# `levels_completed == 0`, so a game that has scored anything is never cut off --
# the point is to stop paying for games that are getting nowhere, not to cap play.
import inference.framework.solver as _solver

# Only classes *defined* in that module: vars() also holds whatever it imported,
# and a match there would be picked first purely by dict order. Requiring exactly
# one match means an upstream change stops the build instead of being resolved by
# an accident of ordering.
_candidates = [_o for _o in vars(_solver).values()
               if isinstance(_o, type) and _o.__module__ == _solver.__name__
               and hasattr(_o, "should_stop") and hasattr(_o, "action_count")]
if len(_candidates) != 1:
    raise SystemExit(f"early-stop: expected exactly one class with should_stop + "
                     f"action_count in inference.framework.solver, found "
                     f"{{[c.__name__ for c in _candidates]}} — upstream changed shape")
_target = _candidates[0]

_original_should_stop = _target.should_stop


def _should_stop_or_hopeless(self) -> bool:
    if _original_should_stop(self):
        return True
    try:
        if int(self.game.current_state.levels_completed) == 0 and \\
                self.action_count >= _STOP_AFTER:
            return True
    except Exception:
        # Never let the extension be the reason a game dies.
        return False
    return False


_target.should_stop = _should_stop_or_hopeless
print(f"duck-delta: early stop armed on {{_target.__name__}} — a game with no level "
      f"cleared stops after {{_STOP_AFTER}} actions, freeing its share of the "
      f"shared vLLM server for the games still playing")
'''


# ------------------------------------------------------------------- model swap
#
# The only lever left that can move the *mean*. Four submissions of the unmodified
# build read 0.98 / 1.14 / 1.11 / 1.23 -- mean 1.115, sigma 0.103 -- and fifty more
# draws project to a best of 1.35, which is 68th. 1.65 is third place and 5.3 sigma
# out, so drawing cannot get there and something has to raise the mean.
#
# Nothing about the model is in the notebook. It is all in `setup_commands.json`
# inside `jeroencottaar/taaf-kaggle-source-share`, which cell 9 reads and executes:
# MODEL_OWNER/MODEL_SLUG, SERVED_MODEL_NAME, the vLLM flags, the sampling
# parameters. That dataset is read-only, so the swap is done by rewriting the
# command text on its way to the shell -- three anchored substitutions, each of
# which raises if its anchor has moved rather than silently running upstream.
#
# Qwen3.6 shipped exactly two open-weight checkpoints: the 27B dense that runs
# today, and 35B-A3B, a sparse MoE with **3B active parameters** claiming parity
# with much larger dense models at a 3B model's speed. Both Apache-2.0. The mirror
# below is the official FP8, 37.5 GB, on a 96 GB Blackwell -- and another ARC-AGI-3
# entrant has already staged the same weights (`zerabytex/qwen-3-6-35b-a3b-fp8`,
# subtitled "RAY100 ARC-AGI-3 offline model artifact"), which is some evidence it
# loads under this stack.
#
# Be clear about what this bet is. Nothing on Kaggle is *smarter* than what we run;
# 35B-A3B is a throughput bet, not a quality one, and the one throughput experiment
# already run -- 25,200 s of wall clock, three times the actions -- scored 0.58.
# What makes this one worth running anyway is that it is measurable: a Phase A push
# costs no submission slot, and tokens/second is a mechanism number, the kind that
# has transferred before when scores did not.
MODEL_SWAPS = {
    # The throughput bet, already qualified and submitted: 3.1x tokens/sec for a
    # 3B-active model, and it scored 0.42 -- the worst of the five. Kept because
    # its Phase A numbers are the reference every later swap is read against.
    "moe": {
        "kernel_dir": "notebooks-duck-moe",
        # Attached through `model_sources`, so Kaggle mounts it at
        # /kaggle/input/<slug>/<framework>/<variation>/<version> -- three levels
        # deeper than a dataset, which is why the path is found by search rather
        # than built from the ref.
        "model_source": "michaelpoluektov/qwen3-6-35b-a3b-fp8/transformers/default/1",
        "served_name": "Qwen/Qwen3.6-35B-A3B-FP8",
        # Long enough to install the wheelhouse, load 37.5 GB, pass the smoke test
        # and play a little. This answers "does it boot and how fast does it
        # generate", not "does it score" -- 9,600 s buys the per-game table, and
        # there is no reason to spend 2.4 h of GPU before the server comes up.
        "budget_s": 1800.0,
    },
    # Qwen3.8, the generation after the one we serve. Found 2026-08-17 by reading
    # the public kernel list rather than the leaderboard: fourteen teams crossed
    # 2.00 between 08-14 and 08-17, and the titles say why --
    # "LB1.71 Qw3.8 27B FP8 Temperature1.0 kv16 xHigh", "ARC-AGI-3: Qwen3.8 27b
    # bf16", "ARC3 Q38 TAAF Baseline". The model-swap note that said nothing
    # available is smarter than what we run was written in the Qwen3.6 era and is
    # simply out of date.
    #
    # A clean drop-in: same publisher as the MoE mirror, same FP8 format, and at
    # 30.9 GB it is lighter than the 35.9 GB we serve. Apache-2.0 on the instance.
    # The risk is the parsers -- setup pins --tool-call-parser qwen3_coder and
    # --reasoning-parser qwen3 -- and if a generation change breaks them the vLLM
    # smoke test fails outright, which is where a wrong guess should surface.
    "qwen3.8": {
        "kernel_dir": "notebooks-duck-q38",
        "model_source": "michaelpoluektov/qwen3-8-27b-fp8/transformers/default/1",
        "served_name": "Qwen/Qwen3.8-27B-FP8",
        "budget_s": 1800.0,
    },
    # The quality bet, and the first one with a product behind it. Qwen3.6-27B
    # with an ARC-AGI-3 synthetic *visual perception* LoRA merged into the base
    # weights (Mario Gemoll, Apache-2.0 on the instance, 54.7 GB BF16).
    #
    # Why this direction at all: measured over the unmodified full-length run, the
    # agent holds a median 1.76x the human's level-1+level-2 action budget and
    # still clears level 2 in only 5 of 25 games. It is decision-starved, not
    # action-starved, so throughput is the currency we have spare and quality is
    # the one we lack.
    #
    # The cost is real and this push is what measures it: BF16 against the FP8 we
    # serve is roughly half the tokens/sec, which would take that 1.76x to ~0.88x.
    # No --dtype or --quantization is pinned in setup_commands.json, so vLLM reads
    # the format from the model config; at the default 0.9 utilisation the 54.7 GB
    # of weights leave ~31 GB of KV cache on the 96 GB card.
    "perception": {
        "kernel_dir": "notebooks-duck-lora",
        "model_source": "mariogemoll/arc-prize-2026-arc-agi-3-vlm/transformers/perception-512/1",
        "served_name": "mariogemoll/ARC-AGI-3-VLM-perception-512",
        # v1 ran at 1800 s and answered "does it boot, and what does it cost":
        # it boots (BF16, no silent fallback to the FP8 snapshot) and it is
        # action-neutral -- 0.57x throughput cancelled by ~0.52x tokens per
        # decision, netting ~1.0x actions against E3's budget-matched Phase A.
        # But 1800 s is mostly wheelhouse install plus a 54.7 GB load, so only
        # ~14 games reached five actions and the medians rest on ~12 actions
        # each. 9,600 s is the length at which the solver's own 7,920 s per-game
        # cap binds rather than the outer budget, which is what buys a per-game
        # table worth reading.
        "budget_s": 9600.0,
    },
}

# Which one this build writes. Default is the MoE so that an unqualified build
# reproduces what is already on Kaggle; set MODEL_SWAP=perception to build the
# quality arm. Only the named kernel_dir is written either way -- the baseline
# notebooks are untouched.
MODEL_SWAP = MODEL_SWAPS[os.environ.get("MODEL_SWAP", "moe")]

# Anchors in the setup command, with what each becomes. Kept as data so the build
# fails loudly on an upstream change instead of pushing an unswapped kernel.
MODEL_SWAP_SUBS = [
    # Kaggle Models do not mount where `resolve_kaggle_dataset_path` looks, so find
    # the weights instead: the one directory under /kaggle/input holding a
    # config.json next to safetensors shards. The wheelhouse has wheels and the
    # source bundle has neither, so the match is unambiguous. Falls back to the
    # original expression, which then trips the existing missing-path check.
    (
        "MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
        "MODEL_PATH = next(\n"
        "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
        "     if any(d.glob('*.safetensors'))),\n"
        "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))",
    ),
    # Only ever a label: it is passed through to LOCAL_ANALYZER_MODEL_ID and
    # INFERENCE_ANALYZER_MODEL and compared against what the server reports back.
    # Nothing branches on its text -- the provider preset is a separate string.
    (
        "SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
        "SERVED_MODEL_NAME = '{served_name}'",
    ),
]

# `--tool-call-parser qwen3_coder` and `--reasoning-parser qwen3` are deliberately
# left alone: 35B-A3B is the same generation as the 27B and shares its chat
# template family. If that is wrong the vLLM smoke test in the setup command fails
# outright, which is exactly where a wrong guess should surface.


# The upstream customization hook is a comment-only cell reserved for exactly this
# kind of tweak, so the edit lands there rather than in the run logic.
HOOK_MARKER = "Make one-off changes to `bm`"
HOOK = f'''\
# Make one-off changes to `bm`, `bm.games`, or `bm.solver` here before the run starts.

if not TRUE_SUBMISSION:
    # Phase A only. The deployed 32,400 s would leave the run idling for hours
    # after every game has stopped; this stops shortly after they do. The solver's
    # own per-game cap is untouched, here and everywhere.
    target.max_runtime_s = {PHASE_A_BUDGET_S!r}
    print(f"duck-delta: Phase A outer budget {{target.max_runtime_s}}s "
          f"(solver per-game cap left at its deployed value)")

# The run-length experiment is gone: raising the solver's per-game wall clock from
# 7,920 s to 25,200 s scored 0.58 where the unmodified harness scored 0.98.
#
# What ships instead is perception. Reading cn04's opening board, a black plug ends
# in two pale blue squares and a purple socket carries two more on its inner edge,
# and pale blue is nowhere else — a person sees the contacts at once. The
# segmentation handed to the model reports four small blue components and nothing
# about their relationship. Measured on all 25 opening boards, the rule "a colour
# used sparingly, in several places, on more than one object" names cn04's contacts
# with no other marker on that board at all, and separately names the three colours
# in sk48 that appear once on the playfield and once in the bottom legend — where
# the legend's left-to-right order (8, 14, 9) differs from the board's top-to-bottom
# order (8, 9, 14), so collecting nearest-first fails and reading the legend does
# not. Both games score zero in the harness's own diagnostic.
#
# The sandbox cannot import anything project-level, which is why segmentation is
# pasted into it as source rather than imported; this pastes the same way, then
# hooks the one line that builds a frame's segmentation.
{{DELTA_BODY}}
'''


CLICK_PROMPT_LINE = (
    "- `click_tally(transitions)` and `click_reading(transitions)` are available "
    "inside the `python` tool. They count, per colour, how many clicks have landed "
    "on that colour and how many of them changed the board at all. A colour marked "
    "`dead` has been clicked several times and has never once done anything, so "
    "clicking it again is a wasted action -- aim at colours with a non-zero rate "
    "instead. This is measured from your own history, not a guess about the game.\n"
)


def click_body() -> str:
    """Paste the click-effect tally into the sandbox and document it."""
    source = (ROOT / "duck_delta" / "click_effect.py").read_text()
    return f"""
import inference.agent.prompts as _prompts2
import inference.agent.python_tool_sandbox as _sandbox2

_CLICK_SOURCE = {json.dumps(source)}
_ANCHOR2 = "HOST_STDOUT = sys.stdout"
_CLICK_PROMPT = {json.dumps(CLICK_PROMPT_LINE)}

_prog = _sandbox2._SANDBOX_BOOTSTRAP
assert _ANCHOR2 in _prog, "duck-delta: sandbox anchor missing (click)"
_n0 = len(_prog)
# Named so the two entry points cannot collide with anything already defined.
_prog = _prog.replace(
    _ANCHOR2,
    _CLICK_SOURCE + chr(10) * 2
    + "click_tally = tally" + chr(10)
    + "click_reading = reading" + chr(10) * 2 + _ANCHOR2, 1)
compile(_prog, "<sandbox>", "exec")
_sandbox2._SANDBOX_BOOTSTRAP = _prog

if _CLICK_PROMPT not in _prompts2.STRUCTURED_RUNTIME_STATE_ADDENDUM:
    _prompts2.STRUCTURED_RUNTIME_STATE_ADDENDUM += _CLICK_PROMPT

print(f"duck-delta: click-effect tally spliced ({{_n0}} -> {{len(_prog)}} bytes)")
"""


MOTION_PROMPT_LINE = (
    "- `motion_reading(transitions)` and `motion_tally(transitions)` are available "
    "inside the `python` tool. They report, per simple action, which colours have "
    "been observed to move and by how much -- measured from your own history, with "
    "the observation count attached. Where a scheme is reported you do not need to "
    "re-derive what the controls do; plan with it. A colour that moves the same way "
    "under every action is flagged separately: that is a counter or timer, not "
    "something you are controlling.\n"
)


def concurrency_body() -> str:
    """Raise the solver's concurrency, to find out whether vLLM saturates at 28.

    Why it is worth knowing. Every game runs its full 7,920 s and ends `gave_up`
    -- no game finishes early, checked across a full-length run -- so the wall
    clock is exactly ceil(games / concurrency) * 7,920. With ~110 games and the
    deployed 32,400 s that is 4 waves and 31,680 s, so the budget is 98% spent
    and total tokens over a run come to roughly

        aggregate_throughput(concurrency) * 32,400

    If throughput rises with concurrency, raising it buys tokens outright, and it
    also cuts the wave count -- which would let the per-game cap go from 7,920 s
    to 10,800 s inside the same budget. If throughput is flat, raising
    concurrency only divides the same tokens more thinly and is worth nothing.

    Locally there are 25 games, so concurrency above 25 changes nothing by
    itself. Pairing it with n_passes makes the extra passes a load generator:
    25 games x 2 = 50 runs, all resident at once when concurrency is 50.

    Phase A only, and gated as such -- not because it would be wrong on a scored
    run but because the hook costs 0.448 there whatever it contains. On a
    measuring run that does not matter.
    """
    return (
        "\nif not TRUE_SUBMISSION:\n"
        "    _was = bm.solver.concurrency\n"
        f"    bm.solver.concurrency = {DELTA_CONCURRENCY}\n"
        "    print(f'duck-delta: solver concurrency {_was} -> "
        "{bm.solver.concurrency}', flush=True)\n"
    )


def unseen_games_body() -> str:
    """Per-game cap for the unseen-games run. The game list itself is NOT set
    here: the run cell rebuilds `bm.games = _offline_games(...)` *after* this
    hook executes, which is exactly how v2 printed "playing 46 unseen games"
    and then played the official 25. The list is redirected by a run-cell
    patch below; only the solver cap survives from here (verified: v2 ran at
    4,500 s/game)."""
    return (
        "\nif not TRUE_SUBMISSION:\n"
        f"    bm.solver.max_runtime_s_per_game = {UNSEEN_CAP_S!r}\n"
        "    print(f'duck-delta: unseen per-game cap "
        "{bm.solver.max_runtime_s_per_game}s', flush=True)\n"
    )


def hook_probe_body() -> str:
    """A1: which half of the null control is guilty -- the import or the print?

    E5 (0.95, byte-identical to v14 but on another kernel) against E3 (0.60, the
    same kernel plus an import and a print) puts the cost squarely on executing
    anything in this hook. It does not say which of the two does it.

    `sys.modules` answers it for free. If `inference.agent.prompts` is already
    loaded when the hook runs, `import` is a dict lookup and cannot plausibly
    cost anything, which leaves the print. If it is absent, this hook is
    triggering a first-time module load at a point the harness never does, and
    the import is the suspect.

    **Phase A only.** Printing is what is under suspicion, so this must never run
    on a scored path -- it is gated by the caller, and it is here to be read out
    of a free diagnostic run, not submitted.
    """
    return '''
def _duck_probe():
    import sys as _sys, time as _t

    _pre = sorted(k for k in _sys.modules if k.startswith("inference"))
    print(f"duck-probe: {len(_pre)} inference.* modules already loaded at hook time")
    for _m in _pre:
        print("duck-probe:   loaded", _m)
    print("duck-probe: prompts already imported?",
          "inference.agent.prompts" in _sys.modules)

    _t0 = _t.perf_counter()
    import inference.agent.prompts as _pnull
    _t1 = _t.perf_counter()
    _added = sorted(set(k for k in _sys.modules if k.startswith("inference"))
                    - set(_pre))
    print(f"duck-probe: the import took {(_t1 - _t0) * 1000:.4f} ms")
    print("duck-probe: modules the import pulled in:", _added or "none")

    _t2 = _t.perf_counter()
    print("duck-probe: timing one print")
    _t3 = _t.perf_counter()
    print(f"duck-probe: a print takes {(_t3 - _t2) * 1000:.4f} ms")
    print("duck-probe: addendum is",
          len(_pnull.STRUCTURED_RUNTIME_STATE_ADDENDUM), "chars")


# Phase A only. The print is one of the two suspects, so this must never execute
# on a scored path -- a probe that costs 0.35 to run would be measuring itself.
if not TRUE_SUBMISSION:
    _duck_probe()
'''


def null_control_body() -> str:
    """Execute on the scored path, and change nothing.

    Deliberately shaped like the other deltas -- an import of a harness module and
    a print -- so that what differs from them is only the absence of any effect.
    """
    return """
import inference.agent.prompts as _pnull

print("duck-delta: null control, prompt addendum is",
      len(_pnull.STRUCTURED_RUNTIME_STATE_ADDENDUM), "chars — nothing changed")
"""


def apply_n_passes(nb) -> int:
    """Rewrite `bm.n_passes` in the run cell. Returns the index of the cell changed.

    This is the only edit outside the customization hook, and it is here because
    the hook cannot do it: the assignment lives in the run cell, which executes
    afterwards and would overwrite anything the hook set.
    """
    target = "bm.n_passes = 1"
    hits = [i for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code" and target in "".join(c["source"])]
    if len(hits) != 1:
        raise SystemExit(f"expected one `{target}`, found {len(hits)} — upstream changed")
    i = hits[0]
    replacement = f"bm.n_passes = {N_PASSES}  # duck-delta: the scorecard takes max over runs"
    if SOLVER_CAP_S is not None:
        replacement += (f"\nbm.solver.max_runtime_s_per_game = {SOLVER_CAP_S!r}"
                        "  # duck-delta: shorter cap buys the second pass")
    text = "".join(nb["cells"][i]["source"]).replace(target, replacement, 1)
    nb["cells"][i]["source"] = text.splitlines(keepends=True)
    return i


def motion_body() -> str:
    """Paste the control-scheme tally into the sandbox and document it."""
    source = (ROOT / "duck_delta" / "motion.py").read_text()
    return f"""
import inference.agent.prompts as _prompts3
import inference.agent.python_tool_sandbox as _sandbox3

_MOTION_SOURCE = {json.dumps(source)}
_ANCHOR3 = "HOST_STDOUT = sys.stdout"
_MOTION_PROMPT = {json.dumps(MOTION_PROMPT_LINE)}

_prog3 = _sandbox3._SANDBOX_BOOTSTRAP
assert _ANCHOR3 in _prog3, "duck-delta: sandbox anchor missing (motion)"
_m0 = len(_prog3)
# Renamed on the way in: `tally` and `reading` are already taken if the
# click-effect delta is also spliced, and a silent shadowing would leave one of
# them answering the other's question.
_MOTION_SOURCE = _MOTION_SOURCE.replace("def tally(", "def motion_tally(").replace(
    "def reading(", "def motion_reading(").replace("table = tally(", "table = motion_tally(")
_prog3 = _prog3.replace(_ANCHOR3, _MOTION_SOURCE + chr(10) * 2 + _ANCHOR3, 1)
compile(_prog3, "<sandbox>", "exec")
_sandbox3._SANDBOX_BOOTSTRAP = _prog3

if _MOTION_PROMPT not in _prompts3.STRUCTURED_RUNTIME_STATE_ADDENDUM:
    _prompts3.STRUCTURED_RUNTIME_STATE_ADDENDUM += _MOTION_PROMPT

print(f"duck-delta: control-scheme tally spliced ({{_m0}} -> {{len(_prog3)}} bytes)")
"""


def temperature_body() -> str:
    """Override the analyzer temperature, in both places it can live.

    `tool_agent` reads it once at import into a module-level constant:

        _LOCAL_ANALYZER_TEMPERATURE = _get_env_float("LOCAL_ANALYZER_TEMPERATURE", 0.6)

    Whether that import has already happened by the time this hook runs depends
    on what unpickling the deployed benchmark drags in, so setting the
    environment variable alone may quietly do nothing. Set both, and assert the
    constant exists — a rename upstream should fail loudly here rather than
    produce a run that looks like an experiment and is the baseline.
    """
    return f"""
import os as _os
import inference.agent.tool_agent as _ta

_TEMP = {ANALYZER_TEMPERATURE!r}
assert hasattr(_ta, "_LOCAL_ANALYZER_TEMPERATURE"), \
    "duck-delta: tool_agent._LOCAL_ANALYZER_TEMPERATURE is gone — check upstream"
_was = _ta._LOCAL_ANALYZER_TEMPERATURE
_os.environ["LOCAL_ANALYZER_TEMPERATURE"] = str(_TEMP)
_ta._LOCAL_ANALYZER_TEMPERATURE = _TEMP
print(f"duck-delta: analyzer temperature {{_was}} -> {{_ta._LOCAL_ANALYZER_TEMPERATURE}}")
"""


def delta_body() -> str:
    """The splice, with `duck_delta/correspondence.py` carried inline.

    Both sources go in as JSON string literals rather than triple-quoted blocks,
    since the module has docstrings of its own and would close the literal early.

    Every step asserts. A silent no-match would upload a notebook that looks
    patched, runs as the unmodified harness, and returns 0.98 a day later looking
    like the idea did not work — so a mismatch raises, and Phase A catches it
    before a slot is spent.
    """
    source = (ROOT / "duck_delta" / "correspondence.py").read_text()
    return f"""
import inference.agent.prompts as _prompts
import inference.agent.python_tool_sandbox as _sandbox

_CORRESPONDENCE = {json.dumps(source)}
_ANCHOR = "HOST_STDOUT = sys.stdout"
_SEG_CALL = "self._segmentation = segment_layer(self._grid, COLOR_CHARS)"
_PROMPT_LINE = {json.dumps(PROMPT_LINE)}

_program = _sandbox._SANDBOX_BOOTSTRAP
assert _ANCHOR in _program, "duck-delta: sandbox anchor missing"
assert _program.count(_SEG_CALL) == 1, "duck-delta: segment_layer call not unique"
_before = len(_program)

_program = _program.replace(_ANCHOR, _CORRESPONDENCE + chr(10) * 2 + _ANCHOR, 1)
_line = next(l for l in _program.splitlines() if l.strip() == _SEG_CALL)
_indent = _line[: len(_line) - len(_line.lstrip())]
_program = _program.replace(
    _line, _line + chr(10) + _indent
    + "self._segmentation['markers'] = find_markers(self._grid)", 1)
compile(_program, "<sandbox>", "exec")
_sandbox._SANDBOX_BOOTSTRAP = _program

if _PROMPT_LINE not in _prompts.STRUCTURED_RUNTIME_STATE_ADDENDUM:
    _prompts.STRUCTURED_RUNTIME_STATE_ADDENDUM += _PROMPT_LINE

print(f"duck-delta: correspondence markers spliced "
      f"({{_before}} -> {{len(_program)}} bytes), prompt extended")
"""


def _patch_cell(nb, needle: str, old: str, new: str, what: str) -> None:
    """Replace `old` with `new` in the one cell containing `needle`."""
    hits = [c for c in nb["cells"]
            if c["cell_type"] == "code" and needle in "".join(c["source"])]
    if len(hits) != 1:
        raise SystemExit(f"model swap: expected 1 cell containing {needle!r}, "
                         f"found {len(hits)} — upstream changed shape")
    text = "".join(hits[0]["source"])
    if old not in text:
        raise SystemExit(f"model swap: anchor for {what} not found — upstream "
                         f"changed shape:\n  {old.splitlines()[0]}")
    hits[0]["source"] = text.replace(old, new, 1).splitlines(keepends=True)


# Substitutions applied to `setup_commands.json` on its way to the shell. This is
# the route E7 proved clean: it scored 1.20 against an untaxed mean of 1.102,
# while anything executed in the customization hook costs 0.448 regardless of what
# it does. So every knob below ships for free, and the hook is off limits.
#
# `tool_agent` reads fourteen settings out of this dict, and the deployed values
# are not the code defaults -- TOOL_STEPS ships at 0 where the code default is 12,
# EVICT_TURNS at 2 where the default is 1. Read the anchors out of the file rather
# than assuming.
SETUP_SUBS = {
    # E7: changes the notebook, changes nothing that runs. The control that
    # established this route.
    "null": [("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
              "SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'"
              "  # duck-delta: null substitution, no behaviour change")],

    # E8: the one tax-invalidated arm that is an env var. Its taxed reading was
    # 0.72; the crude correction puts it near 1.21. Higher temperature is also the
    # variance play -- the board keeps the max, three games carry 96% of the score
    # so n_eff is about 2.3, and with the target far above our mean the spread is
    # worth more than the mean.
    "temperature1.0": [("'LOCAL_ANALYZER_TEMPERATURE': '0.6'",
                        "'LOCAL_ANALYZER_TEMPERATURE': '1.0'")],

    # reasoning_effort is a *new* variable, so the substitution appends it to an
    # existing line rather than replacing a value. It only does anything on a
    # build carrying our bundle fork, which supplies the env -> tool_agent ->
    # payload path; on the stock bundle the variable is read by nobody.
    #
    # "low" rather than the default on purpose for the qualifying run: a value
    # that should visibly change throughput is one we can confirm took effect,
    # where re-sending the default would be indistinguishable from the field
    # being dropped.
    # Qwen3.8 plus reasoning_effort, as one substitution set.
    #
    # Composed here rather than by running MODEL_SWAP and SETUP_SUB together,
    # because both rewrite the same setup-command loop and whichever ran first
    # consumed the anchor -- the combined build came out carrying neither. One
    # list through one wrapper cannot have that failure.
    # Qwen3.8 alone -- the 1.87 configuration, for composing with other changes.
    "q38": [
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
    ],

    # Axis A: the imitation adapter (6,512 human pairs, held-out exact 65.5% /
    # type 74.0% against a 68.2% trivial baseline) served on top of the FP8 base.
    #
    # Routing is client-side: vLLM names the LoRA module 'arc3-sft' as its own
    # model id, so SERVED_MODEL_NAME becomes 'arc3-sft' -- which flows to the
    # smoke-test payload, LOCAL_ANALYZER_MODEL_ID and INFERENCE_ANALYZER_MODEL --
    # while the server's --served-model-name keeps the base under its original
    # name. Every request therefore goes base+adapter, and the boot smoke test
    # exercises the adapter path before any game starts: a failed adapter load is
    # a failed boot, not a silent baseline run.
    #
    # The adapter dataset is found by its adapter_config.json, which cannot
    # collide with the weights search above -- that one keys on config.json (an
    # exact-name rglob) next to safetensors, and the adapter dataset has no
    # config.json.
    "q38_sft": [
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))\n"
         "LORA_PATH = next(\n"
         "    (p.parent for p in Path('/kaggle/input').rglob('adapter_config.json')), None)\n"
         "if LORA_PATH is None:\n"
         "    raise RuntimeError('q38_sft: no adapter_config.json under /kaggle/input -- '\n"
         "                       'is the arc3-sft-lora dataset attached?')"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'arc3-sft'"),
        ("'--served-model-name',\n        SERVED_MODEL_NAME,",
         "'--served-model-name',\n        'Qwen/Qwen3.8-27B-FP8',"),
        ("'--max-model-len',\n        str(VLLM_MAX_MODEL_LEN),\n    ]",
         "'--max-model-len',\n        str(VLLM_MAX_MODEL_LEN),\n"
         "        '--enable-lora',\n"
         "        '--max-lora-rank',\n"
         "        '32',\n"
         "        '--lora-modules',\n"
         "        'arc3-sft=' + str(LORA_PATH),\n    ]"),
    ],

    # Shared by both dflash arms: vLLM's vocab top-k prefers flashinfer's
    # radix kernel, which ships in no cubin and therefore JIT-compiles -- and
    # the image's system nvcc predates Blackwell ("Unsupported gpu
    # architecture 'compute_120f'", v5 death). torch.topk costs ~2x on this
    # one op and needs no compiler. The patch runs after the wheelhouse
    # install, right before the server launches; the print is the audit trail.
    "_dflash_topk_patch": [
        ("def start_vllm_server() -> None:\n    install_vllm_wheelhouse()",
         "def start_vllm_server() -> None:\n    install_vllm_wheelhouse()\n"
         "    _lp = SITE_PACKAGES / 'vllm/model_executor/layers/logits_processor.py'\n"
         "    _lp_src = _lp.read_text(encoding='utf-8')\n"
         "    _lp_needle = 'def _flashinfer_topk() -> Callable[..., tuple[torch.Tensor, torch.Tensor]] | None:'\n"
         "    if '_arc3_topk_patch' not in _lp_src:\n"
         "        if _lp_needle not in _lp_src:\n"
         "            raise RuntimeError('topk patch: anchor missing in logits_processor.py')\n"
         "        _lp.write_text(_lp_src.replace(\n"
         "            _lp_needle,\n"
         "            _lp_needle + chr(10) + '    return None  # _arc3_topk_patch'\n"
         "            ' -- torch.topk; the JIT needs an nvcc this image lacks',\n"
         "            1), encoding='utf-8')\n"
         "        print('taaf.kaggle: flashinfer topk disabled (torch.topk fallback)', flush=True)"),
    ],

    # The ablation for the 29-level result: the dflash arm changed engine
    # (0.19 -> nightly) and added speculation in one step, so this arm is the
    # nightly WITHOUT --speculative-config and without the draft attached.
    # 29 vs this arm's levels = what speculation bought; this arm vs the
    # stock-q38 control's 20 = what the engine bought. Same wheelhouse, same
    # flashinfer taming (arch list, sampler off, topk patch via composition
    # below).
    "q38_nightly": [
        ("WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)",
         "WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)\n"
         "\n"
         "\n"
         "def _dewheel_kaggle_mangling(src):\n"
         "    '''Restore wheel file names Kaggle mangled: devNNNNgHASH -> devNNNN+gHASH.'''\n"
         "    import re as _re\n"
         "    fixed = WORKING_DIR / 'wheelhouse-dewrangled'\n"
         "    fixed.mkdir(parents=True, exist_ok=True)\n"
         "    for f in sorted(src.iterdir()):\n"
         "        name = f.name\n"
         "        if name.endswith('.whl'):\n"
         "            name = _re.sub(r'(\\.dev\\d+)g([0-9a-f]{7,})', r'\\1+g\\2', name)\n"
         "        dst = fixed / name\n"
         "        if not dst.is_symlink() and not dst.exists():\n"
         "            os.symlink(f, dst)\n"
         "    return fixed"),
        ("'--find-links',\n        str(WHEELHOUSE),",
         "'--find-links',\n        str(_dewheel_kaggle_mangling(WHEELHOUSE)),"),
        ("WHEELHOUSE_OWNER = 'driessmit1'",
         "WHEELHOUSE_OWNER = 'tantan0327'"),
        ("WHEELHOUSE_SLUG = 'arc3-vllm-h100-wheelhouse-v3'",
         "WHEELHOUSE_SLUG = 'arc3-vllm-dflash-wheelhouse'"),
        ("STAMP_TEXT = 'vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\\n'",
         "STAMP_TEXT = 'vllm==0.26.1rc1.dev1239+g28b484e5d torch==2.13.0 "
         "flashinfer==0.6.17\\n'"),
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
        ("'VLLM_NO_USAGE_STATS': '1',\n        }",
         "'VLLM_NO_USAGE_STATS': '1',\n"
         "            'FLASHINFER_DISABLE_VERSION_CHECK': '1',\n"
         "            'FLASHINFER_CUDA_ARCH_LIST': '12.0f',\n"
         "            'VLLM_USE_FLASHINFER_SAMPLER': '0',\n        }"),
    ],

    # Axis D: DFlash2 speculative decoding -- the field's lossless-speed
    # mechanism (2.67-3.43x measured by the public kernels that carry it).
    # Needs the nightly-vLLM wheelhouse (the DFlash2 PR is in no release), the
    # q38 FP8 base, and the z-lab draft (3.8 GB bf16, Apache-2.0, mirrored at
    # dangkhoa2016/z-lab-qwen3-8-27b-dflash2).
    #
    # Two models now mount with a config.json next to safetensors, so the
    # weights search must discriminate: the draft's config declares
    # architectures=["DFlash2DraftModel"] and a dflash_config block (verified
    # against the HF original), so "dflash in the config text" is the draft and
    # the other candidate is the base. method='dflash' is the string the
    # nightly's DFlashProposer asserts; num_speculative_tokens=7 matches the
    # draft's block_size of 8 (one query token + 7 drafted).
    "q38_dflash": [
        # Kaggle strips '+' from uploaded file names, so the nightly wheel
        # arrived as vllm-0.26.1rc1.dev1239g28b484e5d-....whl -- a version pip
        # can neither parse nor match against the lock's '+g' pin (found
        # 2026-08-29, the arm's first death after the mount fix). File
        # *contents* are untouched, so requirements.lock still pins correctly;
        # the fix is a symlink farm that restores the original names at run
        # time, no 2 GB re-upload.
        ("WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)",
         "WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)\n"
         "\n"
         "\n"
         "def _dewheel_kaggle_mangling(src):\n"
         "    '''Restore wheel file names Kaggle mangled: devNNNNgHASH -> devNNNN+gHASH.'''\n"
         "    import re as _re\n"
         "    fixed = WORKING_DIR / 'wheelhouse-dewrangled'\n"
         "    fixed.mkdir(parents=True, exist_ok=True)\n"
         "    for f in sorted(src.iterdir()):\n"
         "        name = f.name\n"
         "        if name.endswith('.whl'):\n"
         "            name = _re.sub(r'(\\.dev\\d+)g([0-9a-f]{7,})', r'\\1+g\\2', name)\n"
         "        dst = fixed / name\n"
         "        if not dst.is_symlink() and not dst.exists():\n"
         "            os.symlink(f, dst)\n"
         "    return fixed"),
        ("'--find-links',\n        str(WHEELHOUSE),",
         "'--find-links',\n        str(_dewheel_kaggle_mangling(WHEELHOUSE)),"),
        ("WHEELHOUSE_OWNER = 'driessmit1'",
         "WHEELHOUSE_OWNER = 'tantan0327'"),
        ("WHEELHOUSE_SLUG = 'arc3-vllm-h100-wheelhouse-v3'",
         "WHEELHOUSE_SLUG = 'arc3-vllm-dflash-wheelhouse'"),
        ("STAMP_TEXT = 'vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\\n'",
         "STAMP_TEXT = 'vllm==0.26.1rc1.dev1239+g28b484e5d torch==2.13.0 "
         "flashinfer==0.6.17\\n'"),
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "_cands = [d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "          if any(d.glob('*.safetensors'))]\n"
         "DRAFT_PATH = next(\n"
         "    (d for d in _cands if 'dflash' in (d / 'config.json').read_text().lower()), None)\n"
         "if DRAFT_PATH is None:\n"
         "    raise RuntimeError('q38_dflash: no DFlash draft model mounted under /kaggle/input')\n"
         "MODEL_PATH = next(\n"
         "    (d for d in _cands if d != DRAFT_PATH),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
        # flashinfer-python 0.6.17 (the nightly's hard pin) refuses a cubin
        # package of any other version at import, and PyPI's newest cubin is
        # 0.6.9 -- there is no matching pair to ship. The wheelhouse carries
        # cubin 0.6.9 and this env var (read by the server process via
        # vllm_env) waives the exact-match check; without the package,
        # flashinfer would try to fetch cubins from NVIDIA's artifactory at
        # runtime, which an internet-off kernel cannot do.
        ("'VLLM_NO_USAGE_STATS': '1',\n        }",
         "'VLLM_NO_USAGE_STATS': '1',\n"
         "            'FLASHINFER_DISABLE_VERSION_CHECK': '1',\n"
         # 12.0f preloads TARGET_CUDA_ARCHS directly: the runtime's own
         # detection came up empty on this card (v4 death: an sm75 error
         # on a Blackwell), and the env branch skips detection entirely.
         "            'FLASHINFER_CUDA_ARCH_LIST': '12.0f',\n"
         # v6 death: the sampler is the second JIT trigger; 0 selects
         # vLLM's torch-native sampling path (default is the flashinfer
         # kernels, which ship no cubin for 0.6.17 and cannot compile
         # against this image's pre-Blackwell system nvcc).
         "            'VLLM_USE_FLASHINFER_SAMPLER': '0',\n        }"),
        ("'--max-model-len',\n        str(VLLM_MAX_MODEL_LEN),\n    ]",
         "'--max-model-len',\n        str(VLLM_MAX_MODEL_LEN),\n"
         "        '--speculative-config',\n"
         "        json.dumps({'method': 'dflash', 'model': str(DRAFT_PATH),\n"
         "                    'num_speculative_tokens': 7}),\n    ]"),
    ],

    # The model the 4-point club is converging on, composed with the engine
    # they are NOT using: GPT-OSS-120B (the public LB-9 lineage iterates it on
    # the OLD engine + anim bundle, v14-v18 in two days) on our nightly.
    # Parser names read from the nightly wheel's lazy registry: tool 'openai'
    # -> GptOssToolParser, reasoning 'openai_gptoss' -> GptOssReasoningParser.
    # MXFP4 weights ~63 GB fit the 96 GB card with ~25 GB of KV at 0.9 util.
    "oss120_nightly": [
        ("WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)",
         "WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)\n"
         "\n"
         "\n"
         "def _dewheel_kaggle_mangling(src):\n"
         "    '''Restore wheel file names Kaggle mangled: devNNNNgHASH -> devNNNN+gHASH.'''\n"
         "    import re as _re\n"
         "    fixed = WORKING_DIR / 'wheelhouse-dewrangled'\n"
         "    fixed.mkdir(parents=True, exist_ok=True)\n"
         "    for f in sorted(src.iterdir()):\n"
         "        name = f.name\n"
         "        if name.endswith('.whl'):\n"
         "            name = _re.sub(r'(\\.dev\\d+)g([0-9a-f]{7,})', r'\\1+g\\2', name)\n"
         "        dst = fixed / name\n"
         "        if not dst.is_symlink() and not dst.exists():\n"
         "            os.symlink(f, dst)\n"
         "    return fixed"),
        ("'--find-links',\n        str(WHEELHOUSE),",
         "'--find-links',\n        str(_dewheel_kaggle_mangling(WHEELHOUSE)),"),
        ("WHEELHOUSE_OWNER = 'driessmit1'",
         "WHEELHOUSE_OWNER = 'tantan0327'"),
        ("WHEELHOUSE_SLUG = 'arc3-vllm-h100-wheelhouse-v3'",
         "WHEELHOUSE_SLUG = 'arc3-vllm-dflash-wheelhouse'"),
        ("STAMP_TEXT = 'vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\\n'",
         "STAMP_TEXT = 'vllm==0.26.1rc1.dev1239+g28b484e5d torch==2.13.0 "
         "flashinfer==0.6.17\\n'"),
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'openai/gpt-oss-120b'"),
        # GPT-OSS-120B attention (head_dim 64) triggers flashinfer's xqa kernel,
        # which JIT-compiles and dies on this image's pre-Blackwell nvcc (same
        # family as the topk/sampler JITs). FLASH_ATTN needs no JIT.
        ("'--max-model-len',\n        str(VLLM_MAX_MODEL_LEN),\n    ]",
         "'--max-model-len',\n        str(VLLM_MAX_MODEL_LEN),\n"
         "        '--attention-backend',\n        'TRITON_ATTN',\n    ]"),
        ("'--tool-call-parser',\n        'qwen3_coder',",
         "'--tool-call-parser',\n        'openai',"),
        ("'--reasoning-parser',\n        'qwen3',",
         "'--reasoning-parser',\n        'openai_gptoss',"),
        ("'VLLM_NO_USAGE_STATS': '1',\n        }",
         "'VLLM_NO_USAGE_STATS': '1',\n"
         "            'FLASHINFER_DISABLE_VERSION_CHECK': '1',\n"
         "            'FLASHINFER_CUDA_ARCH_LIST': '12.0f',\n"
         "            'VLLM_USE_FLASHINFER_SAMPLER': '0',\n        }"),
    ],

    # Round-2 adapter on the nightly engine -- the two things that survived
    # their measurements, composed: the engine that moved the board (+7 levels
    # offline, 2.27 banked) serving the adapter trained in the harness's own
    # rendering (tool-call targets, 97.3% parse at 77% of one epoch). No
    # speculation: it measured neutral and is one more part that can break.
    # Adapter arrives via the arc3-sft2-lora dataset (adapter_config.json is
    # the discriminator; the base-model finder keys on config.json and cannot
    # collide).
    "q38_nightly_sft2": [
        ("WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)",
         "WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)\n"
         "\n"
         "\n"
         "def _dewheel_kaggle_mangling(src):\n"
         "    '''Restore wheel file names Kaggle mangled: devNNNNgHASH -> devNNNN+gHASH.'''\n"
         "    import re as _re\n"
         "    fixed = WORKING_DIR / 'wheelhouse-dewrangled'\n"
         "    fixed.mkdir(parents=True, exist_ok=True)\n"
         "    for f in sorted(src.iterdir()):\n"
         "        name = f.name\n"
         "        if name.endswith('.whl'):\n"
         "            name = _re.sub(r'(\\.dev\\d+)g([0-9a-f]{7,})', r'\\1+g\\2', name)\n"
         "        dst = fixed / name\n"
         "        if not dst.is_symlink() and not dst.exists():\n"
         "            os.symlink(f, dst)\n"
         "    return fixed"),
        ("'--find-links',\n        str(WHEELHOUSE),",
         "'--find-links',\n        str(_dewheel_kaggle_mangling(WHEELHOUSE)),"),
        ("WHEELHOUSE_OWNER = 'driessmit1'",
         "WHEELHOUSE_OWNER = 'tantan0327'"),
        ("WHEELHOUSE_SLUG = 'arc3-vllm-h100-wheelhouse-v3'",
         "WHEELHOUSE_SLUG = 'arc3-vllm-dflash-wheelhouse'"),
        ("STAMP_TEXT = 'vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\\n'",
         "STAMP_TEXT = 'vllm==0.26.1rc1.dev1239+g28b484e5d torch==2.13.0 "
         "flashinfer==0.6.17\\n'"),
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))\n"
         "LORA_PATH = next(\n"
         "    (p.parent for p in Path('/kaggle/input').rglob('adapter_config.json')), None)\n"
         "if LORA_PATH is None:\n"
         "    raise RuntimeError('q38_nightly_sft2: no adapter_config.json under /kaggle/input -- '\n"
         "                       'is the arc3-sft2-lora dataset attached?')"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'arc3-sft2'"),
        ("'--served-model-name',\n        SERVED_MODEL_NAME,",
         "'--served-model-name',\n        'Qwen/Qwen3.8-27B-FP8',"),
        ("'VLLM_NO_USAGE_STATS': '1',\n        }",
         "'VLLM_NO_USAGE_STATS': '1',\n"
         "            'FLASHINFER_DISABLE_VERSION_CHECK': '1',\n"
         "            'FLASHINFER_CUDA_ARCH_LIST': '12.0f',\n"
         "            'VLLM_USE_FLASHINFER_SAMPLER': '0',\n        }"),
        ("'--max-model-len',\n        str(VLLM_MAX_MODEL_LEN),\n    ]",
         "'--max-model-len',\n        str(VLLM_MAX_MODEL_LEN),\n"
         "        '--enable-lora',\n"
         "        '--max-lora-rank',\n"
         "        '32',\n"
         "        '--lora-modules',\n"
         "        'arc3-sft2=' + str(LORA_PATH),\n    ]"),
    ],

    # The composite: adapter AND DFlash2 in one server -- depth x speed. One
    # merged list rather than q38_sft + q38_dflash in sequence, because both
    # rewrite MODEL_PATH, SERVED_MODEL_NAME and the cmd tail, and whichever ran
    # first would consume the other's anchor (the exact failure the MODEL_SWAP
    # comment above documents). Verified composable in the nightly: no
    # config-level rejection of lora+speculative anywhere, and the eagle
    # utilities explicitly run the draft without the target's LoRA. The open
    # question the Phase A measures: the draft reads target hidden states that
    # now carry adapter deltas it was not distilled on, so acceptance may drop
    # -- that is a number, not a crash.
    "q38_sft_dflash": [
        # Same Kaggle filename mangling fix as q38_dflash: restore the '+' in
        # the nightly vllm wheel's name via a runtime symlink farm.
        ("WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)",
         "WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)\n"
         "\n"
         "\n"
         "def _dewheel_kaggle_mangling(src):\n"
         "    '''Restore wheel file names Kaggle mangled: devNNNNgHASH -> devNNNN+gHASH.'''\n"
         "    import re as _re\n"
         "    fixed = WORKING_DIR / 'wheelhouse-dewrangled'\n"
         "    fixed.mkdir(parents=True, exist_ok=True)\n"
         "    for f in sorted(src.iterdir()):\n"
         "        name = f.name\n"
         "        if name.endswith('.whl'):\n"
         "            name = _re.sub(r'(\\.dev\\d+)g([0-9a-f]{7,})', r'\\1+g\\2', name)\n"
         "        dst = fixed / name\n"
         "        if not dst.is_symlink() and not dst.exists():\n"
         "            os.symlink(f, dst)\n"
         "    return fixed"),
        ("'--find-links',\n        str(WHEELHOUSE),",
         "'--find-links',\n        str(_dewheel_kaggle_mangling(WHEELHOUSE)),"),
        ("WHEELHOUSE_OWNER = 'driessmit1'",
         "WHEELHOUSE_OWNER = 'tantan0327'"),
        ("WHEELHOUSE_SLUG = 'arc3-vllm-h100-wheelhouse-v3'",
         "WHEELHOUSE_SLUG = 'arc3-vllm-dflash-wheelhouse'"),
        ("STAMP_TEXT = 'vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\\n'",
         "STAMP_TEXT = 'vllm==0.26.1rc1.dev1239+g28b484e5d torch==2.13.0 "
         "flashinfer==0.6.17\\n'"),
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "_cands = [d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "          if any(d.glob('*.safetensors'))]\n"
         "DRAFT_PATH = next(\n"
         "    (d for d in _cands if 'dflash' in (d / 'config.json').read_text().lower()), None)\n"
         "if DRAFT_PATH is None:\n"
         "    raise RuntimeError('q38_sft_dflash: no DFlash draft model mounted under /kaggle/input')\n"
         "MODEL_PATH = next(\n"
         "    (d for d in _cands if d != DRAFT_PATH),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))\n"
         "LORA_PATH = next(\n"
         "    (p.parent for p in Path('/kaggle/input').rglob('adapter_config.json')), None)\n"
         "if LORA_PATH is None:\n"
         "    raise RuntimeError('q38_sft_dflash: no adapter_config.json under /kaggle/input -- '\n"
         "                       'is the arc3-sft-lora dataset attached?')"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'arc3-sft'"),
        ("'--served-model-name',\n        SERVED_MODEL_NAME,",
         "'--served-model-name',\n        'Qwen/Qwen3.8-27B-FP8',"),
        ("'VLLM_NO_USAGE_STATS': '1',\n        }",
         "'VLLM_NO_USAGE_STATS': '1',\n"
         "            'FLASHINFER_DISABLE_VERSION_CHECK': '1',\n"
         # 12.0f preloads TARGET_CUDA_ARCHS directly: the runtime's own
         # detection came up empty on this card (v4 death: an sm75 error
         # on a Blackwell), and the env branch skips detection entirely.
         "            'FLASHINFER_CUDA_ARCH_LIST': '12.0f',\n"
         # v6 death: the sampler is the second JIT trigger; 0 selects
         # vLLM's torch-native sampling path (default is the flashinfer
         # kernels, which ship no cubin for 0.6.17 and cannot compile
         # against this image's pre-Blackwell system nvcc).
         "            'VLLM_USE_FLASHINFER_SAMPLER': '0',\n        }"),
        ("'--max-model-len',\n        str(VLLM_MAX_MODEL_LEN),\n    ]",
         "'--max-model-len',\n        str(VLLM_MAX_MODEL_LEN),\n"
         "        '--enable-lora',\n"
         "        '--max-lora-rank',\n"
         "        '32',\n"
         "        '--lora-modules',\n"
         "        'arc3-sft=' + str(LORA_PATH),\n"
         "        '--speculative-config',\n"
         "        json.dumps({'method': 'dflash', 'model': str(DRAFT_PATH),\n"
         "                    'num_speculative_tokens': 7}),\n    ]"),
    ],

    # Qwen3.8 plus temperature 1.0 -- the public LB1.71 configuration, which sends
    # no reasoning_effort field and takes the model default. The only piece of it
    # our 1.87 arm was missing.
    "q38_temp1.0": [
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
        ("'LOCAL_ANALYZER_TEMPERATURE': '0.6'",
         "'LOCAL_ANALYZER_TEMPERATURE': '1.0'"),
    ],

    "q38_effort_low": [
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
        ("'LOCAL_ANALYZER_TOP_K': '20',",
         "'LOCAL_ANALYZER_TOP_K': '20',\n"
         "    'LOCAL_ANALYZER_REASONING_EFFORT': 'low',"),
    ],

    "effort_low": [("'LOCAL_ANALYZER_TOP_K': '20',",
                    "'LOCAL_ANALYZER_TOP_K': '20',\n"
                    "    'LOCAL_ANALYZER_REASONING_EFFORT': 'low',")],

    # R2. Qwen3.8 -- our banked 1.87 -- plus the one change that stops the harness
    # erasing what the agent worked out. Upstream cuts context to the last 30
    # assistant turns *after* the context trim has already fitted history to
    # ANALYZER_CONTEXT_WINDOW, so it discards history the 32,768-token budget had
    # already paid for, and raising it cannot overflow.
    #
    # One action is one turn in our runs and the median game is 132 turns, so the
    # agent sees a median 23% of its own game and 8% on the two longest. Of the
    # 1,763 repeats of a (board, action) it had already tried, 42.6% were of
    # attempts already evicted. 100 turns would have shown it 92.4% against 57.4%,
    # and the 99th percentile gap is 195, so 100 buys most of the effect without
    # asking the context trim to work hardest.
    #
    # Needs BUNDLE_FORK=1: the constant is source-level, not an upstream env var.
    "q38_history100": [
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
        ("'LOCAL_ANALYZER_TOP_K': '20',",
         "'LOCAL_ANALYZER_TOP_K': '20',\n"
         "    'LOCAL_ANALYZER_HISTORY_TURNS': '100',"),
    ],

    # R1. Qwen3.8 -- our banked 1.87 -- plus the stall guard, which is aimed at the
    # one thing measurement actually pins on us: against a random policy on the same
    # 25 games the agent clears at least one level 11/25 against random's 2/25, and
    # both win zero. It is 5.5x chance at starting and equal to chance at finishing.
    #
    # A level the agent completed never took more than 150 actions; a level it
    # stalled on burned a median 122 and up to 397. So at 150 actions without an
    # advance the guard arms and refuses any (level, board, action) already tried
    # there -- provably re-entry into a cycle, since the masked playfield is
    # deterministic to 93.4% agreement. Replayed over the agent's own recorded runs
    # it arms on 11 levels across 9 of 25 games and blocks 13.0% of all actions.
    #
    # It blocks rather than tells, because every modification this project shipped
    # that informed the model lost -- the click-effect tally cut wasted clicks
    # 86% -> 15% and scored 0.69.
    #
    # Needs BUNDLE_FORK=1: source-level, and gated off by default.
    "q38_stall150": [
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
        ("'LOCAL_ANALYZER_TOP_K': '20',",
         "'LOCAL_ANALYZER_TOP_K': '20',\n"
         "    'LOCAL_ANALYZER_STALL_GUARD': '1',\n"
         "    'LOCAL_ANALYZER_STALL_ACTIONS': '150',"),
    ],

    # T2 Phase A: Qwen3.8 with the banking gate on. Only meaningful on the anim
    # fork (the gate lives in its solver) and only interesting where wins happen,
    # i.e. combined with UNSEEN_GAMES=1.
    "q38_banking": [
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
        ("'LOCAL_ANALYZER_TOP_K': '20',",
         "'LOCAL_ANALYZER_TOP_K': '20',\n"
         "    'LOCAL_ANALYZER_BANKING': '1',"),
    ],

    # T1 rung 2: keep xhigh thinking, emit several actions per thought. effort_low
    # doubled actions (+82%) and lost a third of the levels; the batch hint aims
    # for the actions without touching the thinking.
    "q38_batch": [
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
        ("'LOCAL_ANALYZER_TOP_K': '20',",
         "'LOCAL_ANALYZER_TOP_K': '20',\n"
         "    'LOCAL_ANALYZER_BATCH_HINT': '1',"),
    ],

    # T1 fallback rung: if the batch hint fails, 'medium' sits between xhigh
    # (48.9 actions/game) and 'low' (88.8 actions/game, quality collapsed).
    "q38_effort_medium": [
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
        ("'LOCAL_ANALYZER_TOP_K': '20',",
         "'LOCAL_ANALYZER_TOP_K': '20',\n"
         "    'LOCAL_ANALYZER_REASONING_EFFORT': 'medium',"),
    ],

    # The composite: stall85 (the one lever aimed at deep progress, never yet
    # measured -- 150 never fired and 85 has not run) plus banking (engine-only,
    # can only add). Thinking untouched: both throughput rungs showed more
    # actions monotonically costs levels, so deliberation stays at xhigh.
    "q38_composite": [
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
        ("'LOCAL_ANALYZER_TOP_K': '20',",
         "'LOCAL_ANALYZER_TOP_K': '20',\n"
         "    'LOCAL_ANALYZER_STALL_GUARD': '1',\n"
         "    'LOCAL_ANALYZER_BANKING': '1',"),
    ],

    # O2: keep what the agent worked out — goal/world/action models survive level
    # transitions and game overs (only the plan is falsified) — plus the
    # mechanical evidence ledger the harness writes into the prompt every turn.
    "q38_o2": [
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
        ("'LOCAL_ANALYZER_TOP_K': '20',",
         "'LOCAL_ANALYZER_TOP_K': '20',\n"
         "    'LOCAL_ANALYZER_KEEP_KNOWLEDGE': '1',\n"
         "    'LOCAL_ANALYZER_LEDGER': '1',"),
    ],

    # S2: the acting half of the frontier map -- goto_frontier() in the sandbox.
    "q38_goto": [
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
        ("'LOCAL_ANALYZER_TOP_K': '20',",
         "'LOCAL_ANALYZER_TOP_K': '20',\n"
         "    'LOCAL_ANALYZER_FRONTIER_GOTO': '1',"),
    ],

    # Axis B: the experiment protocol appended to the per-turn instructions.
    "q38_protocol": [
        ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
         "MODEL_PATH = next(\n"
         "    (d for d in sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')})\n"
         "     if any(d.glob('*.safetensors'))),\n"
         "    resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG))"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
        ("'LOCAL_ANALYZER_TOP_K': '20',",
         "'LOCAL_ANALYZER_TOP_K': '20',\n"
         "    'LOCAL_ANALYZER_PROTOCOL': '1',"),
    ],
}

# Which substitution set this build ships, or None for the plain baseline.
SETUP_SUB = os.environ.get("SETUP_SUB") or None

# Point the notebook at our own fork of the MIT bundle instead of upstream's.
# Needed for anything source-level -- the customization hook costs 0.448 whatever
# it contains, and the setup route only reaches env vars and the vLLM launch.
# The bundle is located by rglob-ing /kaggle/input for taaf-kaggle-bundle.json, so
# the original must be *replaced* rather than added: two markers make the search
# ambiguous.
BUNDLE_FORK = os.environ.get("BUNDLE_FORK") or None


# The topk patch rides with every nightly-vLLM arm; composed here so a fix
# cannot silently apply to one arm and not the others.
SETUP_SUBS["q38_dflash"] = SETUP_SUBS["q38_dflash"] + SETUP_SUBS["_dflash_topk_patch"]
SETUP_SUBS["q38_sft_dflash"] = (SETUP_SUBS["q38_sft_dflash"]
                                + SETUP_SUBS["_dflash_topk_patch"])
SETUP_SUBS["q38_nightly"] = SETUP_SUBS["q38_nightly"] + SETUP_SUBS["_dflash_topk_patch"]
SETUP_SUBS["oss120_nightly"] = (SETUP_SUBS["oss120_nightly"]
                                + SETUP_SUBS["_dflash_topk_patch"])
# The retained-reasoning lead (OpenAI: preserving what the model learned
# tripled their public-set score) re-priced on the engine that carries our
# board result: nightly + a 100-turn history window. The old null (1.43) was
# a single draw on the 0.19 engine and binds nothing.
SETUP_SUBS["q38_nightly_h100"] = SETUP_SUBS["q38_nightly"] + [
    ("'LOCAL_ANALYZER_TOP_K': '20',",
     "'LOCAL_ANALYZER_TOP_K': '20',\n"
     "    'LOCAL_ANALYZER_HISTORY_TURNS': '100',"),
]
# v8: the world-model contract, gated on. Needs BUNDLE_FORK (the kit, the
# planner and the contract live in the fork's sandbox/tool_agent patches).
SETUP_SUBS["q38_nightly_v8"] = SETUP_SUBS["q38_nightly"] + [
    ("'LOCAL_ANALYZER_TOP_K': '20',",
     "'LOCAL_ANALYZER_TOP_K': '20',\n"
     "    'ARC3_V8_WORLDMODEL': '1',"),
]
# D3.1: the first v8 run crashed every game with "Analyzer did not return a
# result" -- 58 chat turns overflowed the 31,744-token context and the loop
# yielded before executing. v8 keeps its state in the sandbox (predict/verify
# live there), so a long chat history is redundant with the design; a short
# window fixes the overflow AND matches the architecture. HISTORY_TURNS lives
# in the fork, so this needs BUNDLE_FORK like the base v8 sub.
SETUP_SUBS["q38_nightly_v8_h6"] = SETUP_SUBS["q38_nightly"] + [
    ("'LOCAL_ANALYZER_TOP_K': '20',",
     "'LOCAL_ANALYZER_TOP_K': '20',\n"
     "    'ARC3_V8_WORLDMODEL': '1',\n"
     "    'LOCAL_ANALYZER_HISTORY_TURNS': '6',"),
]
SETUP_SUBS["q38_nightly_sft2"] = (SETUP_SUBS["q38_nightly_sft2"]
                                 + SETUP_SUBS["_dflash_topk_patch"])


def apply_setup_subs(nb, subs, label: str) -> None:
    """Rewrite `setup_commands.json` between reading it and running it.

    The bundle is read-only, so the edit happens in the notebook, in the same cell
    and the same way the model swap does it. Every anchor is checked at run time
    and a miss stops the run: the failure to avoid is a build that looks patched,
    executes as the baseline, and returns a number a day later that means nothing.
    """
    loop = 'for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):'
    patched = (
        f"_SETUP_SUBS = {json.dumps(subs, indent=4)}\n"
        "\n"
        "\n"
        "def _apply_setup_subs(command: str) -> str:\n"
        "    if 'PYSETUP' not in command:\n"
        "        return command\n"
        "    for old, new in _SETUP_SUBS:\n"
        "        if old not in command:\n"
        "            raise SystemExit(\n"
        "                'setup subs: anchor missing from setup_commands.json, '\n"
        "                'refusing to run something that is not what was built: ' + old)\n"
        "        command = command.replace(old, new, 1)\n"
        f"    print('taaf.kaggle: setup subs applied ({label})', flush=True)\n"
        "    return command\n"
        "\n"
        "\n"
        + loop.replace(
            'json.loads((BUNDLE_DIR / "setup_commands.json").read_text())',
            '[_apply_setup_subs(c) for c in '
            'json.loads((BUNDLE_DIR / "setup_commands.json").read_text())]')
    )
    _patch_cell(nb, "setup_commands.json", loop, patched, "the setup command loop")


def apply_model_swap(nb) -> None:
    """Point the harness at a different model, without touching the solver.

    Three edits, all outside the solver: drop the old weights from the attached
    inputs, rewrite the setup command on its way to the shell, and set this
    build's Phase A budget. Every one is anchored, and a missing anchor stops the
    build -- the failure mode to avoid is pushing something that looks swapped and
    silently runs the 27B.
    """
    swap = MODEL_SWAP
    subs = [(old, new.replace("{served_name}", swap["served_name"]))
            for old, new in MODEL_SWAP_SUBS]

    # 1. The old model dataset is no longer attached, so take it out of the list the
    #    notebook maps to mount points. Leaving it would map it to a path that does
    #    not exist -- harmless here, but it would also leave a second candidate for
    #    the weights search below if it ever were attached again.
    _patch_cell(
        nb, "DATASET_SOURCES = [",
        ', "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]', "]",
        "DATASET_SOURCES")

    # 2. `setup_commands.json` lives in a read-only dataset, so the substitutions
    #    happen between reading it and running it.
    loop = 'for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):'
    swapped = (
        f"_MODEL_SWAP_SUBS = {json.dumps(subs, indent=4)}\n"
        "\n"
        "\n"
        "def _swap_model(command: str) -> str:\n"
        '    """Retarget the vLLM launch, or stop the run if the command has moved."""\n'
        "    if 'PYSETUP' not in command:\n"
        "        return command\n"
        "    for old, new in _MODEL_SWAP_SUBS:\n"
        "        if old not in command:\n"
        "            raise SystemExit(\n"
        "                'model swap: anchor missing from setup_commands.json, refusing to '\n"
        "                'run the wrong model: ' + old.splitlines()[0])\n"
        "        command = command.replace(old, new, 1)\n"
        f"    print('taaf.kaggle: model swapped to {swap['served_name']}', flush=True)\n"
        "    return command\n"
        "\n"
        "\n"
        + loop.replace(
            'json.loads((BUNDLE_DIR / "setup_commands.json").read_text())',
            '[_swap_model(c) for c in '
            'json.loads((BUNDLE_DIR / "setup_commands.json").read_text())]')
    )
    _patch_cell(nb, "setup_commands.json", loop, swapped, "the setup command loop")

    # 3. This build's own Phase A length, independent of the baseline's.
    if swap["budget_s"] != PHASE_A_BUDGET_S:
        _patch_cell(nb, HOOK_MARKER,
                    f"target.max_runtime_s = {PHASE_A_BUDGET_S!r}",
                    f"target.max_runtime_s = {swap['budget_s']!r}",
                    "the Phase A budget")


def harden_wheels_path(nb) -> None:
    """Stop trusting where Kaggle mounts the competition wheels this week.

    2026-08-29: all three Phase A kernels died at the upstream runtime-install
    cell because its hardcoded path
    (/kaggle/input/competitions/<slug>/arc_agi_3_wheels) stopped existing --
    the mount layout moved out from under it. The patch checks the two layouts
    Kaggle has used, then falls back to searching for the directory and finally
    for the arc_agi wheel file itself, and fails with a message that names the
    problem instead of pip's "no matching distribution".
    """
    old = (
        '        "--find-links",\n'
        '        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",\n'
        '        "arc-agi",\n'
        "    ],\n"
        "    stdout=subprocess.DEVNULL,\n"
        ")"
    )
    new = (
        '        "--find-links",\n'
        "        str(ARC_WHEELS),\n"
        '        "arc-agi",\n'
        "    ],\n"
        "    stdout=subprocess.DEVNULL,\n"
        ")"
    )
    resolver = (
        "_wheel_cands = [\n"
        '    Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"),\n'
        '    Path("/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"),\n'
        "]\n"
        "ARC_WHEELS = next((p for p in _wheel_cands if p.is_dir()), None) or next(\n"
        '    (p for p in Path("/kaggle/input").rglob("arc_agi_3_wheels") if p.is_dir()), None) or next(\n'
        '    (p.parent for p in Path("/kaggle/input").rglob("arc_agi-*.whl")), None)\n'
        "if ARC_WHEELS is None:\n"
        '    raise RuntimeError("arc_agi wheels not found anywhere under /kaggle/input")\n'
        "subprocess.check_call("
    )
    # Two cells derive paths from the moved mount: the install cell, and the
    # run cell whose competition_env_files (the offline games) hangs off the
    # same hardcoded parent. Both are keyed to ARC_WHEELS, resolved once in the
    # install cell -- notebook cells share one namespace, so the run cell reads
    # it directly. Needles are lines unique to each cell, because the bare
    # directory name appears in both.
    install_needle = ('"/kaggle/input/competitions/arc-prize-2026-arc-agi-3/'
                      'arc_agi_3_wheels",')
    _patch_cell(nb, install_needle, "subprocess.check_call(", resolver,
                "the wheels path resolver")
    _patch_cell(nb, install_needle, old, new, "the wheels find-links path")
    _patch_cell(
        nb, "competition_env_files = str(Path(",
        'competition_env_files = str(Path("/kaggle/input/competitions/'
        'arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")',
        'competition_env_files = str(ARC_WHEELS.parent / "environment_files")',
        "the competition env files path")


def main() -> None:
    """Patch the vendored upstream notebook and write the kernel we push."""
    nb = json.loads(UPSTREAM.read_text())
    harden_wheels_path(nb)

    notice_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": NOTICE.splitlines(keepends=True),
    }

    hooks = [
        i for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "code" and HOOK_MARKER in "".join(c["source"])
    ]
    if len(hooks) != 1:
        raise SystemExit(
            f"expected exactly one customization hook cell, found {len(hooks)} — "
            "upstream changed shape; re-read it before trusting this build"
        )
    bodies = []
    if DELTA_CORRESPONDENCE:
        bodies.append(delta_body())
    if DELTA_CLICK_EFFECT:
        bodies.append(click_body())
    if DELTA_NULL_CONTROL:
        bodies.append(null_control_body())
    if DELTA_HOOK_PROBE:
        bodies.append(hook_probe_body())
    if DELTA_CONCURRENCY:
        bodies.append(concurrency_body())
    if DELTA_MOTION:
        bodies.append(motion_body())
    if DELTA_EARLY_STOP is not None:
        bodies.append(early_stop_body())
    if ANALYZER_TEMPERATURE is not None:
        bodies.append(temperature_body())
    if UNSEEN_GAMES:
        bodies.append(unseen_games_body())
    hook = HOOK.replace("{DELTA_BODY}", "\n".join(bodies)
                        or "# no delta — this build is upstream\n")
    nb["cells"][hooks[0]]["source"] = hook.splitlines(keepends=True)
    passes_cell = apply_n_passes(nb) if N_PASSES is not None else None
    nb["cells"].insert(0, notice_cell)

    # The baseline kernel is drawn from every day and must stay exactly as the
    # last plain build left it. Any experimental flag therefore writes ONLY the B
    # notebook -- writing both is how the unseen-games hook ended up sitting in
    # the baseline file, invisible to git because that path is gitignored.
    # MODEL_SWAP deliberately not in this list: it is a constant (the moe kernel
    # always builds) and it writes only its own kernel_dir, never OUT.
    experimental = bool(bodies) or (N_PASSES is not None) or \
        bool(BUNDLE_FORK) or bool(SETUP_SUB)
    text = json.dumps(nb, indent=1) + "\n"
    targets = (OUT_B,) if experimental else (OUT, OUT_B)
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    if experimental:
        print(f"  baseline {OUT.relative_to(ROOT)} left untouched (experimental build)")

    # The swapped build is a third kernel, not a replacement: the baseline is drawn
    # from every day and must stay exactly as it is while this is being qualified.
    if MODEL_SWAP:
        moe = json.loads(text)
        apply_model_swap(moe)
        moe_path = ROOT / MODEL_SWAP["kernel_dir"] / "duck.ipynb"
        moe_path.parent.mkdir(parents=True, exist_ok=True)
        moe_path.write_text(json.dumps(moe, indent=1) + "\n")
        print(f"wrote {moe_path.relative_to(ROOT)}  "
              f"(model -> {MODEL_SWAP['served_name']}, "
              f"Phase A {MODEL_SWAP['budget_s']:.0f}s)")

    # After the model-swap copy, which builds from `text` and needs the unpatched
    # setup loop to find its own anchor.
    if BUNDLE_FORK:
        _patch_cell(nb, "DATASET_SOURCES = [",
                    '"jeroencottaar/taaf-kaggle-source-share"',
                    f'"{BUNDLE_FORK}"',
                    "the bundle dataset")

    if os.environ.get("DROP_MODEL_DATASET"):
        if True:
            # A swapped model arrives through model_sources, so the old weights
            # must leave this list too -- otherwise it maps to a mount that does
            # not exist, and worse, gives the config.json search a second
            # candidate if it is ever attached again.
            _patch_cell(nb, "DATASET_SOURCES = [",
                        ', "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]', "]",
                        "the old model dataset")
        print(f"  bundle fork: {BUNDLE_FORK} (upstream bundle replaced, not added)")

    if SETUP_SUB:
        apply_setup_subs(nb, SETUP_SUBS[SETUP_SUB], SETUP_SUB)
        print(f"  setup subs applied: {SETUP_SUB} (hook cell untouched)")

    if DEPTH_GAMES:
        _ids = sorted(set(DEPTH_GAMES.split(",")))
        _cap = float(os.environ.get("SOLVER_CAP_S") or 31680.0)
        _patch_cell(
            nb, "bm.games = _offline_games(competition_env_files)",
            "    bm.games = _offline_games(competition_env_files)",
            "    bm.games = _offline_games(competition_env_files)\n"
            f"    _depth = {set(_ids)!r}\n"
            "    bm.games = [g for g in bm.games"
            " if g.env_name.split('-')[0] in _depth]\n"
            f"    bm.solver.max_runtime_s_per_game = {_cap!r}\n"
            "    print(f'duck-delta: DEPTH mode, {len(bm.games)} games at "
            "{bm.solver.max_runtime_s_per_game}s each', flush=True)",
            "the depth game list")
        print(f"  depth mode: {len(_ids)} games at {_cap:.0f}s each")

    if UNSEEN_GAMES:
        # The run cell rebuilds bm.games after the customization hook, so the
        # only place a different game list can survive is inside that rebuild.
        _patch_cell(
            nb, "bm.games = _offline_games(competition_env_files)",
            "    bm.games = _offline_games(competition_env_files)",
            "    _unseen_cands = ["
            "Path('/kaggle/input/arc3-unseen-small-games/environment_files'), "
            "Path('/kaggle/input/datasets/tantan0327/arc3-unseen-small-games/environment_files')]\n"
            "    _unseen_root = next((c for c in _unseen_cands if c.exists()), None)\n"
            "    if _unseen_root is None:\n"
            "        raise RuntimeError('unseen dataset not mounted: ' + str(_unseen_cands))\n"
            "    bm.games = _offline_games(str(_unseen_root))\n"
            "    print(f'duck-delta: playing {len(bm.games)} unseen games from "
            "{_unseen_root}', flush=True)",
            "the offline game list")
        print("  unseen games: run cell redirected to the unseen dataset")

    # Both of the above mutate `nb` after the first write, so write back once
    # here. Keeping the write inside one of the branches meant a fork-only build
    # patched the notebook in memory and shipped the unmodified file.
    #
    # Only OUT_B, for the reason the MODEL_SWAP branch above already states: OUT
    # is the baseline kernel the daily draw runs against, and it must stay exactly
    # as it is while anything else is being qualified. Writing both left an
    # experimental configuration sitting in the baseline directory, one
    # `cycle.py push --target duck` away from becoming the baseline -- and
    # invisible to `git status`, because that path is gitignored.
    if BUNDLE_FORK or SETUP_SUB:
        OUT_B.write_text(json.dumps(nb, indent=1) + "\n")

    code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(nb['cells'])} cells, {code} code)")
    print(f"  Phase A budget : {PHASE_A_BUDGET_S:.0f}s (upstream 32400s)")
    names = (["NULL CONTROL"] if DELTA_NULL_CONTROL else []) + \
            (["HOOK PROBE (Phase A only)"] if DELTA_HOOK_PROBE else []) + \
            (["correspondence"] if DELTA_CORRESPONDENCE else []) + \
            (["click-effect tally"] if DELTA_CLICK_EFFECT else []) + \
            (["control scheme"] if DELTA_MOTION else []) + \
            ([f"early stop @{DELTA_EARLY_STOP}"] if DELTA_EARLY_STOP is not None else []) + \
            ([f"temperature {ANALYZER_TEMPERATURE}"] if ANALYZER_TEMPERATURE is not None else [])
    if N_PASSES is not None:
        names.append(f"n_passes {N_PASSES} (run cell {passes_cell})")
    print("  scored-run delta: " + (", ".join(names) or "none — upstream"))


if __name__ == "__main__":
    main()
