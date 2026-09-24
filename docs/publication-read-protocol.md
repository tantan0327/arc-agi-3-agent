# Publication read protocol — the 72 hours after a leader publishes

Written 2026-09-22 for the 9/30 Milestone-2 open-sourcing (an eligibility
precondition: every M2 claimant must publish by then) and for any later
publication. It is a checklist with gates, in the order the answers are needed.
Every "verified" below means a primary source was read; anything else is written
down as **unverified**.

## 0. Where the publications appear (check daily from 9/28)

- `scratchpad/m2-publication-watch.log` — the launchd watcher
  (`scripts/watch_m2_publications.sh`, `scripts/launchd/com.tantan.arc3-m2-watch.plist`)
  snapshots the top-10 and the newest public kernels.
- Kaggle: the leaders' kernels made public; the competition discussion tab
  ("Milestone 2 solution" posts); linked GitHub repositories; arcprize.org.
- Kaggle pages are a SPA: `WebFetch`/`curl` get only a title. Use the API
  (`kaggle kernels pull`, `kaggle datasets`/`models` commands) and clone GitHub.

## 1. Read (target: 6 hours from publication). One row per leader.

| question | where the answer is | write down |
|---|---|---|
| model, quantisation, size on disk | kernel metadata `model_sources` / `dataset_sources`; `kaggle models instances get owner/slug/framework/variation` (bytes, licence) | name, GiB, licence |
| **fits one RTX PRO 6000 (94.97 GiB) with KV room?** | weights GiB + KV: Flash-Next NVFP4 sits at 78–82 GiB and leaves 5 GiB KV (~105k tokens); larger KV OOMs; a 27B FP8 is ~31 GB, a 27B BF16 54.7 GB | GiB, KV headroom |
| serving stack and its licence chain | the setup cell: image digest, vLLM version/wheelhouse dataset, launch flags (`--tool-call-parser`, `--reasoning-parser`, utilisation, max-num-seqs) | stack, versions, licences |
| harness lineage | does it attach `jeroencottaar/taaf-kaggle-source-share` (MIT) or a fork (e.g. `thtennant/taaf-kaggle-source-share-fork`, CC0)? which symbols exist: `ToolAgent._build_user_prompt`, `run_sandboxed_python`, `_HarnessGameSession._execute_action`, `step_env`, `_compact_action_result`, `_summarized_knowledge_lines` | Duck-lineage yes/no; hook symbols present |
| how history, reasoning and the world model are carried | prompt builder + compaction: frames kept, reasoning retained, note/world-model text, images on/off | the retention policy |
| learned component? | a CNN/world model/LoRA/RL policy; is it trained at test time, on what, for how long; does training fit the 30 h weekly quota | kind, test-time cost |
| per-game clock and action policy | `max_runtime_s_per_game`, batching, early stop | seconds, batch size |
| the author's own number | their stated LB score AND their offline number if given; single draws are ±0.5 on the board (our floor: 14 draws, sd 0.47) | score, n |
| what is NOT open | closed weights, private data, a training run they do not ship | the missing piece |

**Gate A (read):** a method with closed weights, a licence outside the rule, or a
training run we cannot afford is skipped — go to the next leader. The rule
(arcprize.org, verified 8/5): our own code CC0/MIT-0; third-party code under any
share-permitting OSS licence (Apache-2.0, GPLv3 named as sufficient).

## 2. Verify the licence chain (target: 24 hours)

- Datasets: `GET /api/v1/datasets/view/{owner}/{slug}` → `licenseName` (the
  `kaggle datasets metadata` CLI returns None for everything — not a check).
- Kaggle Models: `kaggle models instances get …` → licence field.
- Docker image: match the digest against Docker Hub (the flash-next check is the
  template: official vLLM image, digest matched, weights Qwen Community 1.0).
- Code: LICENSE files and pyproject fields in every attached source bundle; the
  Duck source is MIT with no LICENSE file — the attribution cell must travel.

**Gate B (licence):** every link verified or the method is not boarded.

## 3. Stand it up as-is (target: 48 hours)

1. `kaggle kernels pull` → a fresh kernel dir under our account; **byte-identical**
   notebook and the same attached datasets/models/image (the flash-next fork
   scored 2.80 first time this way).
2. Boot smoke: Phase A 1,800 s (`scripts/build_duck_notebook.py` for Duck-lineage
   notebooks; the setup-substitution route, never the customisation hook). Read
   the log for the vLLM smoke test, the served model name, tokens/s.
3. Full-length offline run: 25 games × 7,920 s (2.4 h GPU) via
   `scripts/build_probe.py <name> --seconds 7920` if the notebook is Duck-lineage
   (its post-setup cell is where env overrides and patches go — setup persists
   `LOCAL_ANALYZER_*`/`MULTIMODAL_*` into `os.environ`, so cell-3 overrides are
   silently overwritten). Compare with `scripts/compare_probes.py`.
4. Read it against **our** instrument, not theirs: twelve full runs of one
   vehicle span 32–41 levels (sd ~3), score sd ~1. Per-game flips decide, totals do
   not. Verify any injected text reached the model by grepping `transcripts/*.txt`.

**Gate C (boot + reproduce):** boots under our account and plays; the offline
number is within our noise band of the author's if they gave one.

## 4. Board it (target: 72 hours)

- Ask the user (every modified/new board submission needs explicit approval).
- `scripts/arm_candidate.sh --arm` / `scripts/switch_draw_to_candidate.sh` retarget
  `scripts/daily_draw.sh` (kernel id + version + note); `scripts/validate_candidate.sh`
  checks the kernel version exists and the note matches. One draw per UTC day;
  three draws give a family mean (±0.28 at sd 0.47).
- Keep if the 3-draw mean beats the current vehicle's mean; the board keeps the
  max of all plays, so a higher-mean family is the only thing worth drawing.

## 5. Then graft — one change per build, three draws each

Take the grafts from `graft-inventory.md` in the order their evidence ranks
them; each goes on through the same post-setup cell (Duck-lineage) or through
the interface the inventory lists; each is measured against the method's own
as-is control on our stack before it is boarded.

## The clock

| T+ | done |
|---|---|
| 6 h | §1 table for every leader; gate A |
| 24 h | §2 licence chain; gate B; kernel pulled and pushed as-is (Phase A) |
| 48 h | §3 full-length run read per game; gate C |
| 72 h | §4 boarded with approval; draws start |
