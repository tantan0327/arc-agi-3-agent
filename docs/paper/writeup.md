# The agent starts like an agent and stops like a coin flip
## A computational-neuroscience dissection of why humans solve ARC-AGI-3 and an LLM agent does not

*Paper Track writeup (v4). Linked submission: [tantan0327/arc3-franzen-dprime](https://www.kaggle.com/code/tantan0327/arc3-franzen-dprime) v1, submission ID 57160861 (public 33.55; per-draw mean 27.2). Paper (PDF), scripts, artefacts: [github.com/tantan0327/arc-agi-3-agent](https://github.com/tantan0327/arc-agi-3-agent) (MIT-0; Appendix B maps claims to scripts). Data and CPU reproduction: [kaggle.com/datasets/tantan0327/arc3-paper-artefacts](https://www.kaggle.com/datasets/tantan0327/arc3-paper-artefacts) (CC0).*

## Abstract

We measure where an LLM agent fails on ARC-AGI-3 faculty by faculty, as a systems neuroscientist dissects a behaviour: measure each faculty against ground truth, intervene on it, see what moves. Against 6,512 recorded human actions, random and clairvoyant policies, and the game engine, on the two open harnesses that won the competition's milestones, we find that perception, motion and action efficiency are not the gap: the agent is near-human on contingency and super-human on efficiency (term 1.05 of a 1.15 cap over 775 cleared levels). Six brain-inspired learning mechanisms attached to the harness changed nothing; the one lever acting only in the hidden run (slot allocation) is bounded at +4 by a calibrated simulator. The deficit is the inference of a level's objective as a multi-step causal chain: a code-fitted core-knowledge theory predicts 72% of human movement and 76% of goal signatures yet clears 0 of 37 levels, and an open model authors the needed abstraction on 4 of 22 games. The leaderboard is a max-statistic with per-draw sd 2.7–3.5, and three engine/harness defects are documented.

## Introduction

ARC-AGI-3 scores Relative Human Action Efficiency, how few actions an agent needs to clear the levels of unseen games against a human baseline, on environments that "only leverage Core Knowledge priors". Humans solve 100%; open LLM agents clear a few levels per game. The benchmark is a cognitive-science artefact by construction (Chollet 2019; Spelke & Kinzler 2007). Humans learn a game in minutes because they bring structured hypothesis spaces about objects and agents and infer, from one or two interactions, its dynamics and its objective (theory-based RL, Tsividis et al. 2021). We ask which parts of that solution an LLM agent has, with a measurement for each.

## Prior work

Chollet (2019) and the ARC-AGI-3 Technical Report define the metric and the admissible priors; Spelke & Kinzler (2007) separate objects from agents; Dubey et al. (2018) show the object prior is the one whose removal costs humans most; Lake et al. (2017) and Tsividis et al. (2021) give the theory-based account we test. On ARC-AGI-3, Tufa Labs' Duck harness and Franzen's patched version (the milestone winners we instrument) are LLM tool-use loops; Rudakov et al. (2025) explore state graphs training-free; Rodionov (2026) induces executable world models with frontier models; ARC Prize's Astra analysis documents in-context symbolic world modelling at $19–26k per game; Affectify's published slot-priority retune is the allocation lever we bound. The brain-inspired mechanisms we attach (predictive coding, prioritised replay, gated model-based control, resource-rational allocation) are new as harness interventions on this benchmark.

## Approach

**The submission.** The linked entry is Franzen's Milestone-2 notebook byte for byte (Duck harness, MIT; his Apache-2.0 patch; Qwen3.8-Flash-Next W4A16 with an MTP drafter; Pennyroyal SGLang) plus one cell of ours: a clean-room CC0 implementation of the published D' slot priority, which decides which waiting game gets a freed model slot in the 110-game, ten-slot hidden run.

**The instruments.** (1) The official human replays, replayed 22/22 in sync on the local engine. (2) A random policy at twice the human budget, and a clairvoyant policy where a bound is needed. (3) The engine, so every claim about transitions is executed, not read. (4) A noise model of the leaderboard (fixed configurations drawn daily, best-of-N arithmetic) and a hidden tax: executing anything in the harness's customisation hook costs 0.448 of score (p = 0.0079), so our arms use a verified clean route.

**The dissection (algorithm level).** For each faculty of the human solution (the ledger's rows) we (a) measure it in the agent against the human data and (b) intervene on it as one change to the harness, read against the vehicle's band. Three instruments are new: a theory-induction test (fit the simplest core-knowledge theory by counting: the thing that moved last is the agent and translates per key unless a colour that blocked before is in the way; predict each human transition before seeing it); a goal-signature census (the colour the agent enters at the level-completing action); and a theory-based planner that plays levels 2+ in the real engine, re-planning on surprise. For allocation, an event-driven simulator of the shared-server run built from 213 real trajectories (tokens ÷ 61 tokens/s), validated on every demonstration run and a contended run, calibrated to the board, with a clairvoyant oracle as the bound.

## Results

**Leaderboard.** Linked entry: best public draw 33.55 (rank ~90), per-draw mean 27.2 over nine scored draws; the byte-identical copy draws 26.3. Each team displays its best draw: 76 one-shot copies of the winning notebook scored mean 26.8, max 36.1; the retune advertised at 31–32 draws 27.2; a fixed configuration has sd 2.7–3.5 at this scale.

**Public evaluation (25 public games).** The ledger:

| faculty | measurement | verdict |
|---|---|---|
| contingency | 94% of actions change the board (humans 98.7%) | human-level |
| efficiency | 775 cleared levels: actions/baseline 0.67; term 1.05 vs cap 1.15 | super-human; score is depth only |
| state sufficiency | 380 repeated states: 43% agree as rendered, 90% with the discarded animation frames | repairable |
| object vs agent | no single object moves in 40% of games; theory as object 72%, as colour 46% | partial |
| dynamics induction | a counter predicts 72% of 4,083 human direction moves; the model re-derives mechanics in 6–10 turns per level | slow |
| predictive coding | predict-then-check adopted (474 hits/140 misses); levels 42/40 vs band 44/50/40 | null |
| replay | retained solver replayed by the harness: 2 of 45 replays cleared a level | null |
| act-every-turn; exploration; context; yield | levels in band; tokens per level unchanged | null |
| objective | modal colour entered at 76% of completions (7 games 100%); planner on it: **0 of 37 levels** | **the deficit** |
| abstraction authoring | open model: planner-grade abstraction on 4 of 22 games in 437 attempts | the same deficit |
| objective prior; code-owned motion | 11 demo runs (archetype prior, held-out regularities, census, motion helpers passive and forced): 41–50 vs band 44/50/40; tokens/level 18–22k in all | null; text priors not adopted |
| allocation | oracle +4 over base priority, +2 over D'; D' +1 per draw; throughput linear to the 10-stream cap (285→761 tok/s) | small; throughput binds |

**Reading.** The agent is 5.5× chance at starting a game and exactly chance at finishing one; it does not fail by being slow but by continuing. Objectives are multi-step causal chains (collect, then exit; toggle, then pass): the modal colour is their signature, not their content, and neither a counter, a search, a written prior nor any attached mechanism supplies the chain. The games the demonstration excludes are slow, not stuck: given 60 minutes each, eighteen cleared 69 levels, so patience, not insight, is what the retune buys.

## Conclusion

At Marr's computational level the task is joint inference of dynamics and objective under core-knowledge priors; at the algorithmic level the local parts are cheap and the objective's causal chain is a structured search the model performs in context, slowly, which no attached module performs for it; at the implementational level the agent pays in tokens on one GPU, so allocation and throughput are the run's economics, and both are bounded here. What transfers: the leaderboard noise model (per-draw sd and expected best-of-k from one configuration drawn N times, which any best-of-draws board needs); the allocation simulator (the ceiling of any priority policy for a shared-server run, from per-action token traces and a throughput calibration); the theory-induction and goal-signature tests (how much of the human policy a counting theory reproduces on any action-scored interactive benchmark, and the deficit left); and a null ledger with a script per entry. The simulator also prices the prescription: cutting tokens per action by 20–40%, what a code-owned theory of motion removes, is worth +3 to +6 on this board, more than the allocation lever itself. What to build: a theory-based harness where code owns what counters predict and tokens go to the objective's chain, with the gate we found, and three predictions: levels per token rise with the share of transitions the code theory predicts; a model that states the objective chain after one completion clears the next level in human-like counts and one that does not never does; no slot reallocation lifts this chassis past ~32. We did not solve the benchmark; we located the faculty that does.

*Text CC-BY 4.0; code MIT-0/CC0.*
