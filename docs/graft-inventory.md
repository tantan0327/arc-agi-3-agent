# Graft inventory — what we can add to a published method, with interfaces

Written 2026-09-22. One row per graft: what it is, where it lives, how it
attaches, what is verified, what was measured. "Delivered" means the text or
behaviour was verified to reach the model in transcripts (two runs in September
were void because an injection point was dead code — `_describe_last_outcome`;
the user prompt is rendered by `ToolAgent._build_user_prompt`). Measurements are
full-length offline runs (25 games × 7,920 s) read per game against the
twelve-run band (levels 32–41) unless a board number is given.

## A. Harness grafts (Duck-lineage: attach through the post-setup cell built by `scripts/build_probe.py --patch`)

| graft | file | hooks it needs | status | measured | port cost |
|---|---|---|---|---|---|
| **Note backfill** — the carried world-model note is filled from the assistant's retained reasoning when a turn ends in a tool call with no prose (this model thinks, then calls the tool, so the note was silently empty for whole games) | `scripts/patches/note_from_reasoning.py` | `_summarized_knowledge_lines`, `_extract_scientist_note` | fired (bp35 0→31, m0r0 0→19 prose-less turns carrying a note) | offline `notefull` 38 levels; sk48 L1, m0r0 L2, sb26 L2 cleared for the first time in six runs. **Board (candidate B): 2.77, 3.55, 3.67, 2.62 — mean 3.15 vs floor 2.87 (n=14, sd 0.47)** | low on Duck-lineage; elsewhere: wherever the harness keeps a cross-turn note |
| **Animation channel** — intermediate frames of an action (the engine returns several; the harness kept only the last) summarised into the action result and one prompt sentence | `scripts/patches/animation_channel.py` | `_HarnessGameSession._execute_action`, `step_env` (batch), `_compact_action_result`, `_summarize_step_sequence`, `_build_user_prompt` | delivered (730 appearances, 25/25 games) | `animfull` v2 40 levels, every game in band — **null** | medium |
| **Perception facts oracle** — the physics planner learns from the agent's own moves which body the keys move, its position, step per key, blocking/deadly colours, and writes a "Mechanical facts" block into the prompt | `scripts/build_facts_patch.py` (inlines `duck_delta/physics_planner.py`) | `ToolAgent.analyze`, `_build_user_prompt`, `load_runtime_state` | delivered (636 appearances, 16/25 games) | `factsfull` v2 37 levels, in band — **null**; tool timeouts 5→1 from the 90 s timeout | medium |
| **goto tools** — the same planner as helpers inside the python tool: `body()`, `goto(row,col)`, `goto_colour(c)`; plan with measured rules, execute, re-plan on disagreement | `duck_delta/goto_tool.py` (stdlib-only prelude), `scripts/build_goto_patch.py` → `scratchpad/goto_patch.py` | `tool_agent.run_sandboxed_python` (prelude prepended), `_build_user_prompt` (helper text) | delivered; real-sandbox e2e `scripts/test_goto_tool.py` 5/9 rigid targets at human action counts | `gotofull` 35 levels, in band; `goto` called 13× in 5 games, almost never at level 2 — **null as an offered tool**. **9/22 fix:** refused edges (a move refused with nothing visible in the way is remembered per position, judged on the body not the board, chunks only over proven ground; `scripts/test_goto_refused_edge.py` passes) | low on any python-tool harness |
| Physics planner as a **player** (harness acts when the fit is trusted) | `duck_delta/physics_planner.py` `PhysicsStage`, `scripts/build_physics_patch.py` | `ToolAgent.analyze` | gate (`min_verified=20`, miss ≤ 10%) never earned in the real action budget | `physfull` 41 levels — zero overlap with where it acted; variance. **closed** | — |
| Tool timeout 90 s | env `LOCAL_ANALYZER_TOOL_TIMEOUT` in the post-setup cell | — | verified (timeouts 5→1) | no level effect; hygiene, keep | trivial |
| Stall amnesia, strip reasoning, no-op guard, context/KV/max-num-seqs/image-off/output-cap/history-cap | `scripts/patches/stall_amnesia.py`, `strip_reasoning.py`, `duck_delta/noop_guard.py`, env arms | — | measured | null or negative — **closed** (numbers in `TASKS.md`) | — |

**Precondition for all of A:** the method is Duck-lineage (`taaf-kaggle-source-share`
or a fork) with these symbols. If not, each graft's minimal interface is: note
backfill = access to the model's last reasoning + the carried note; animation =
the engine's per-action frame list + the action-result dict; facts/goto = the
frame history as int grids + a python tool with `action()`; all = the prompt
builder.

## B. Data and offline assets (harness-agnostic)

| asset | where | use | status |
|---|---|---|---|
| Human replays — 22 games, 6,512 actions, every level of every game; replay offline 22/22 | `vendor/taaf-kaggle-source/src/re-arc-3/re_arc/dsl/official_human_replays/`, `scripts/replay_humans.py` | ground truth for goals; `base_actions_per_level`; click-target priors (humans click small rare blobs; two games target the background); SFT/STaR data | verified. As LoRA SFT on Qwen3.8-27B (`arc3-sft-lora` v7, `arc3-sft2-lora`): null-to-negative on the board |
| STaR self-success set (414 efficient pairs) | `scripts/build_star_dataset.py` | training data | built; unmeasured on the board |
| Offline game pool — 190 games, 165 never run, 44 tagged easy | `vendor/taaf-kaggle-source/src/re-arc-3` | CPU-side checks; RL environments | verified count |
| Abstraction ladder / scorer | `scripts/measure_abstraction_ladder.py`, `scripts/score_abstraction.py`, `scripts/gpb_reference_abstractions.py` | novelty 98.3% (board) → 37.9% (object census): reusable structure exists once variables are chosen | offline only |
| Determinism ladder + world-model kit + BFS planner | `duck_delta/worldmodel_kit.py`, `worldmodel_search.py`, `scripts/measure_agency.py` | repairs exact-frame determinism (43% raw → 99% ex-sc25) for any executable-world-model method | offline-proven (planner re-solves 85/85 recorded levels); never boarded |
| Physics planner (rigid bodies) | `duck_delta/physics_planner.py`, tests `scripts/test_physics_planner.py`, `test_physics_stage.py` | dynamics oracle for a learned agent | offline-verified |

## C. Measurement instrument

| item | where | what it gives |
|---|---|---|
| Probe builder | `scripts/build_probe.py <name> [--patch file] [--seconds N]` → `notebooks-probe-<name>/`, kernel `tantan0327/arc3-probe-<name>` | a full-length or 30-minute run with env overrides + patch in the post-setup cell (index 10 on the floor notebook — after setup, which overwrites cell-3 env) |
| Readers | `scripts/compare_probes.py`, `scripts/compare_serving.py`, `scripts/probe_queue.sh` | levels/actions/calls/TTFT/queue/cache-hit per run; two GPU sessions at a time |
| Noise band | twelve full runs of one vehicle: levels 32–41 (sd ~3), score sd ~1; board floor sd 0.47 | decide by per-game flips on the stuck games, never by totals |
| Delivery check | `grep` the injected phrase in `transcripts/*.txt`; count helper CALLS in tool code, not prelude injections | catches dead injection points |
| Efficiency pricing | `actions_per_level` / `base_actions_per_level` in `benchmark.json` | perfect efficiency at the same depth is worth +10–15%; the gap to the top is depth |

## D. Serving receptacles

| path | notebook / kernel | state |
|---|---|---|
| flash-next stack (Qwen3.8-Flash-Next NVFP4, pinned dev vLLM + PLE offload, official image digest verified) | `notebooks-flashnext-asis/` (`arc3-flashnext-asis` v1, the floor), `notebooks-probe-notefull/` (candidate B) | drawn daily; 78–82 GiB of 94.97; KV 5 GiB; larger KV OOMs |
| stock vLLM 0.19 wheelhouse (`driessmit1/arc3-vllm-h100-wheelhouse-v3`), 4-constant model binding (`MODEL_SWAPS` in `scripts/build_duck_notebook.py`) | `notebooks-duck-q38/` (`arc3-duck-qwen3-8-27b`; Qwen3.8-27B FP8 30.9 GB) | **boots (9/22, v5, unchanged 8/19 notebook):** wheelhouse installs (vllm 0.19.0, torch 2.10.0), served as `Qwen/Qwen3.8-27B-FP8`, 403 tokens/s generation, 25 games / 316 actions / 3 levels in the 13 min Phase A left after a ~6 min install+load; output in `scratchpad/q38-smoke_out/` |
| nightly vLLM wheelhouse (0.26.1rc1 + DFlash2/MTP) | dataset `tantan0327/arc3-vllm-dflash-wheelhouse`; profiles `q38_nightly`, `q38_dflash` | booted in August |

## E. Board automation

`scripts/daily_draw.sh` (launchd, every 3 h from 09:07 JST; one submission per UTC
day; reconciles scores into `SUBMISSIONS.md`); `scripts/arm_candidate.sh`,
`scripts/switch_draw_to_candidate.sh`, `scripts/validate_candidate.{sh,py}`; revert
= the three lines at the top of `daily_draw.sh`. Every modified board submission
needs the user's approval.

## Order of application on a new method (by evidence)

1. Note backfill (board-positive on our vehicle).
2. goto tools with the refused-edge fix — only if the method's model reaches for
   tools at level 2 (ours did not); check adoption in transcripts before judging.
3. Animation channel, perception facts (delivered, null on our model; a different
   model may read them differently — one run each, per-game readout).
4. Human replays / STaR into the method's training data, if it trains.
5. Determinism kit under any executable-world-model method.
