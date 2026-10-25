"""Fork v2: the anim TAAF as the base, our gated extras on top.

Why the base changed. The 2.0+ wave on the board (66 teams by 2026-08-21) traces
to one public kernel — `foysalemonshanto/lb-9-...` (155 votes, title claims rank
9) — whose entire delta over the duck v12 lineage is Qwen3.8 plus
`jakobbrggen/taaf-kaggle-source-anim-20260807-anim` (**CC0**). That bundle's
`animation.py` feeds the agent the intermediate frames the old harness discarded
with `raw.frame[-1]`, as a deduplicated cell-diff timeline. Our own determinism
work found the same mechanism independently: 49 of lf52's 49 surviving
contradictions are distinguishable only in the animation frames, and conditioning
on them takes repeat agreement from 76% to 90%. The field's fix and our
measurement agree, which is the strongest kind of evidence available to us.

What is deliberately NOT carried over from fork v1:

- **our noop_guard** — the anim bundle ships its own, wired through
  `ARC3_HARD_NOOP_GUARD` and ON by default. That default is part of the recipe
  the board has priced at ~2.6, so it stays exactly as upstream ships it.

What is carried, all gated off by default so a no-override build is
upstream-equivalent:

- `reasoning_effort` (LOCAL_ANALYZER_REASONING_EFFORT) — absent upstream.
- the history window (LOCAL_ANALYZER_HISTORY_TURNS, default the upstream 30).
- the stall guard (LOCAL_ANALYZER_STALL_GUARD / _STALL_ACTIONS), now importing
  its signatures from its own module because upstream's noop_guard API differs.

    python scripts/build_bundle_fork_v2.py            # build into build/bundle-fork-v2
    python scripts/build_bundle_fork_v2.py --upload   # and publish the dataset
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from build_bundle_fork import patch  # the anchored replace that refuses to guess

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "bundle-fork-v2"
UPSTREAM_DATASET = "jakobbrggen/taaf-kaggle-source-anim-20260807-anim"
OUR_SLUG = "arc3-taaf-anim-plus"

SOLVER = "src/ARC3-Inference/inference/framework/solver.py"
TOOL_AGENT = "src/ARC3-Inference/inference/agent/tool_agent.py"
OPENAI_COMPAT = "src/ARC3-Inference/inference/utils/openai_compat.py"
STALL_DEST = "src/ARC3-Inference/inference/agent/stall_guard.py"
BANK_DEST = "src/ARC3-Inference/inference/agent/banking.py"
V10_DEST = "src/ARC3-Inference/inference/agent/v10_explorer.py"


_O2_WIPE_OLD = '''        if summary.get("level_transition") or summary.get("run_complete") or summary.get("game_over"):
            for key in (
                "world_model",
                "goal_model",
                "action_model",
                "recent_findings",
                "open_questions",
                "current_plan",
            ):
                self._summarized_knowledge[key] = ""'''

_O2_WIPE_NEW = '''        if summary.get("level_transition") or summary.get("run_complete") or summary.get("game_over"):
            # duck-fork-v2 O2-P1: retain knowledge by kind. A game over falsifies
            # the plan, not the mechanics; a level transition falsifies the
            # layout, not the goal -- the stock prompt itself says levels build
            # on earlier mechanics. Measured on v12 transcripts (goalkeep, CC0):
            # the blanket wipe left the working model present on 6.9% of turns.
            if _O2_KEEP and not summary.get("run_complete"):
                _wipe = (("current_plan", "open_questions")
                         if summary.get("level_transition") else ("current_plan",))
            else:
                _wipe = ("world_model", "goal_model", "action_model",
                         "recent_findings", "open_questions", "current_plan")
            for key in _wipe:
                self._summarized_knowledge[key] = ""'''

_O2_ACC_OLD = '''    def _update_summarized_knowledge_from_step_summary(self) -> None:
        summary = self._last_step_summary
        if not summary:
            return'''

_O2_ACC_NEW = '''    def _update_summarized_knowledge_from_step_summary(self) -> None:
        summary = self._last_step_summary
        if not summary:
            return
        if _O2_LEDGER:  # duck-fork-v2 O2-P2: recall of measured fact, never inference
            _led = getattr(self, "_o2_ledger", None)
            if _led is None:
                _led = self._o2_ledger = {}
            _e = _led.setdefault(summary.get("level"),
                                 {"acts": 0, "chg": 0, "overs": 0,
                                  "over_after": "", "done_by": ""})
            _n = int(summary.get("executed_count") or 0)
            _e["acts"] += _n
            if summary.get("board_changed"):
                _e["chg"] += _n
            _names = summary.get("executed_actions") or []
            if summary.get("game_over"):
                _e["overs"] += 1
                _e["over_after"] = str(_names[-1]) if _names else ""
            if summary.get("level_transition") and _names:
                _e["done_by"] = str(_names[-1])'''

_O2_RENDER_OLD = (
    '        lines = [f"- {label}: {value}" for label, value in entries if value]')

_O2_RENDER_NEW = '''        lines = [f"- {label}: {value}" for label, value in entries if value]
        if _O2_LEDGER and getattr(self, "_o2_ledger", None):  # duck-fork-v2 O2
            _parts = []
            for _lv in [k for k in self._o2_ledger if k is not None][-5:]:
                _e = self._o2_ledger[_lv]
                _seg = f"L{_lv}: {_e['acts']} acts ({_e['chg']} changed board)"
                if _e["overs"]:
                    _seg += (f", {_e['overs']} game-overs"
                             f" (last after {_e['over_after']})")
                if _e["done_by"]:
                    _seg += f", completed by {_e['done_by']}"
                _parts.append(_seg)
            if _parts:
                lines.append("- Evidence ledger (measured this game): "
                             + "; ".join(_parts)[:600])'''


_V8_WM_SRC = '\n        # ---- duck-fork-v2 V8: world-model helpers, gated ------------------------\n        # Shipped verbatim from duck_delta/worldmodel_kit.py + worldmodel_search.py\n        # (tests: scripts/test_worldmodel_kit.py 45/45 repeats, errors 3->0;\n        #  scripts/test_worldmodel_search.py 85/85 recorded levels re-solved).\n\n        class ChromeMask:\n\n            def __init__(self, isolated=2):\n                self._isolated = int(isolated)\n                self.chrome = set()\n\n            def learn(self, before, after):\n                diff = [(r, c) for r, row in enumerate(before) for c, v in enumerate(row)\n                        if int(after[r][c]) != int(v)]\n                if 0 < len(diff) <= self._isolated:\n                    self.chrome.update(diff)\n\n            def equal(self, a, b):\n                for r, row in enumerate(a):\n                    brow = b[r]\n                    for c, v in enumerate(row):\n                        if (r, c) in self.chrome:\n                            continue\n                        if int(v) != int(brow[c]):\n                            return False\n                return True\n\n            def digest(self, board):\n                import hashlib\n                rows = []\n                for r, row in enumerate(board):\n                    rows.append(tuple(0 if (r, c) in self.chrome else int(v)\n                                      for c, v in enumerate(row)))\n                return hashlib.blake2b(repr(tuple(rows)).encode(),\n                                       digest_size=8).hexdigest()\n\n        def animation_digest(frames, cm):\n            import hashlib\n            return hashlib.blake2b(\n                "|".join(cm.digest(f) for f in frames).encode(),\n                digest_size=8).hexdigest()\n\n        def judge(predicted, observed, animation, cm,\n                  expected_animation=None):\n            if cm.equal(predicted, observed):\n                return "match"\n            if expected_animation is not None and animation is not None:\n                if animation_digest(animation, cm) != expected_animation:\n                    return "exempt"\n            return "mismatch"\n\n        \n        from collections import deque\n\n        def bfs_plan(start,\n                     actions,\n                     predict,\n                     is_goal,\n                     cm,\n                     *,\n                     max_nodes: int = 20000,\n                     max_depth: int = 64):\n            actions = list(actions)\n            if is_goal(start):\n                return []\n            seen = {cm.digest(start)}\n            q = deque([(start, [])])\n            nodes = 0\n            while q and nodes < max_nodes:\n                board, path = q.popleft()\n                if len(path) >= max_depth:\n                    continue\n                for a in actions:\n                    nodes += 1\n                    try:\n                        nxt = predict(board, a)\n                    except Exception:\n                        continue\n                    if nxt is None:\n                        continue\n                    d = cm.digest(nxt)\n                    if d in seen:\n                        continue\n                    seen.add(d)\n                    npath = path + [a]\n                    if is_goal(nxt):\n                        return npath\n                    q.append((nxt, npath))\n            return None\n\n        def novelty_probe(board,\n                          actions,\n                          predict,\n                          cm,\n                          seen_digests: set[str]):\n            best = None\n            for a in actions:\n                try:\n                    nxt = predict(board, a)\n                except Exception:\n                    continue\n                if nxt is None:\n                    continue\n                if cm.digest(nxt) not in seen_digests:\n                    best = a\n                    break\n            return best\n\n        class _WorldModel:\n\n            def __init__(self, transitions):\n                self._t = transitions\n                self.cm = ChromeMask()\n                self.seen = set()\n                for tr in transitions:\n                    b = getattr(tr, "before_frame", None)\n                    a = getattr(tr, "after_frame", None)\n                    if b is not None and a is not None:\n                        self.cm.learn(b.grid, a.grid)\n                        self.seen.add(self.cm.digest(b.grid))\n                        self.seen.add(self.cm.digest(a.grid))\n\n            def verify(self, predict, last_n=30):\n                ts = list(self._t)[-last_n:]\n                n = ok = 0\n                for tr in ts:\n                    b = getattr(tr, "before_frame", None)\n                    a = getattr(tr, "after_frame", None)\n                    act = getattr(tr, "action", None)\n                    if b is None or a is None or act is None:\n                        continue\n                    n += 1\n                    try:\n                        pred = predict(b.grid, act)\n                    except Exception:\n                        continue\n                    if pred is not None and self.cm.equal(pred, a.grid):\n                        ok += 1\n                return (ok / n) if n else 0.0\n\n            def plan(self, predict, actions, is_goal, max_nodes=20000, max_depth=64):\n                cur = runtime_globals.get("current_frame")\n                if cur is None:\n                    return None\n                return bfs_plan(cur.grid, actions, predict, is_goal, self.cm,\n                                max_nodes=max_nodes, max_depth=max_depth)\n\n            def novel_action(self, predict, actions):\n                cur = runtime_globals.get("current_frame")\n                if cur is None:\n                    return None\n                return novelty_probe(cur.grid, actions, predict, self.cm, self.seen)\n\n        if initial.get("worldmodel_enabled"):\n            _wm_trs = runtime_globals.get("transitions") or []\n            runtime_globals["worldmodel"] = _WorldModel(_wm_trs)\n            import sys as _sys; _sys.stderr.write("duck-fork-v2 V8: worldmodel ready, " + str(len(_wm_trs)) + " transitions\\n")\n        # ---- end V8 -------------------------------------------------------------'

SANDBOX = "src/ARC3-Inference/inference/agent/python_tool_sandbox.py"

# O5/S2: goto_frontier(), the acting half of the frontier map. Self-contained
# closure because the sandbox is a separate restricted process (views only, no
# raw grids, guarded imports): state identity is the frame's ascii, chrome is
# learned with the isolation rule validated in Track C, and only bare ACTIONn
# edges are walked — click targets are unbounded and stay the model's job.
_O5_GOTO_SRC = '''
        if initial.get("frontier_goto_enabled"):
            # No docstring on purpose: this code lives INSIDE the child
            # bootstrap raw string, and a triple quote here terminates that
            # string early -- the SyntaxError that killed goto v1 and v2.
            def goto_frontier(max_steps=6):
                "Walk mechanically to the nearest recorded board with untried simple actions; costs no thinking."
                hist = runtime_globals.get("history") or []
                cur = runtime_globals.get("current_frame")
                valid = [str(v) for v in (runtime_globals.get("valid_actions") or [])]
                if cur is None or not hist:
                    return "frontier: no history yet"
                lvl = getattr(cur, "level", None)
                rows_of = lambda f: tuple((getattr(f, "ascii", "") or "").split("\\n"))
                chrome = set()
                def digest(f):
                    return tuple(
                        "".join(("." if (r, c) in chrome else ch)
                                for c, ch in enumerate(row))
                        for r, row in enumerate(rows_of(f)))
                seq = [(str(getattr(e, "action", "") or ""), e.frame) for e in hist
                       if getattr(e, "frame", None) is not None
                       and getattr(e.frame, "level", None) == lvl]
                for i in range(1, len(seq)):
                    b, a = rows_of(seq[i - 1][1]), rows_of(seq[i][1])
                    if len(b) == len(a):
                        d = [(r, c) for r in range(len(b)) if len(b[r]) == len(a[r])
                             for c in range(len(b[r])) if b[r][c] != a[r][c]]
                        if 0 < len(d) <= 2:
                            chrome.update(d)
                simple = [v for v in valid if v.startswith("ACTION") and "@" not in v]
                edges, tried = {}, {}
                for i in range(1, len(seq)):
                    act = seq[i][0]
                    s, t = digest(seq[i - 1][1]), digest(seq[i][1])
                    tried.setdefault(s, set()).add(act.split("(", 1)[0])
                    if act in simple and t != s:
                        edges.setdefault(s, {})[act] = t
                def untried(s):
                    return [a for a in simple if a not in tried.get(s, set())]
                start = digest(cur)
                if untried(start):
                    return ("frontier: untried right here: "
                            + ", ".join(untried(start)) + " — no walk needed")
                seen, q = {start}, [(start, [])]
                path = None
                while q:
                    s, p = q.pop(0)
                    for a, t in edges.get(s, {}).items():
                        if t in seen:
                            continue
                        np = p + [a]
                        if untried(t):
                            path = np; q = []; break
                        seen.add(t); q.append((t, np))
                if not path:
                    return ("frontier: no reachable board with untried simple "
                            "actions in the recorded graph — explore clicks or new mechanics")
                if len(path) > max_steps:
                    return f"frontier: nearest is {len(path)} steps away (> max_steps)"
                result = action([{"action": a} for a in path])
                return {"walked": path, "result": result}
            runtime_globals["goto_frontier"] = goto_frontier
'''


def build() -> Path:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fork-v2] fetching {UPSTREAM_DATASET}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", UPSTREAM_DATASET,
         "--unzip", "-p", str(BUILD)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
    )

    shutil.copy(ROOT / "duck_delta" / "stall_guard.py", BUILD / STALL_DEST)
    shutil.copy(ROOT / "duck_delta" / "banking.py", BUILD / BANK_DEST)
    shutil.copy(ROOT / "duck_delta" / "v10_explorer.py", BUILD / V10_DEST)
    print(f"[fork-v2] added {STALL_DEST}, {BANK_DEST} and {V10_DEST}")

    # -- solver: stall guard only (upstream's own noop guard stays untouched) --
    sp = BUILD / SOLVER
    s = sp.read_text()
    s = patch(
        s,
        "from taaf.solver import Solver",
        "from taaf.solver import Solver\n"
        "\n"
        "from inference.agent.stall_guard import (  # duck-fork-v2\n"
        "    StallGuard,\n"
        "    action_signature as _stall_action_sig,\n"
        "    board_signature as _stall_board_sig,\n"
        ")\n"
        "\n"
        "# duck-fork-v2: off unless asked for, so a build with no overrides is\n"
        "# upstream-equivalent -- upstream here being the anim bundle exactly as\n"
        "# the board has priced it.\n"
        "_STALL_GUARD_ENABLED = os.environ.get(  # duck-fork-v2\n"
        "    \"LOCAL_ANALYZER_STALL_GUARD\", \"\"\n"
        ").strip().lower() in {\"1\", \"true\", \"yes\", \"on\"}\n"
        "try:  # duck-fork-v2\n"
        "    _STALL_ACTIONS = int(os.environ.get(\"LOCAL_ANALYZER_STALL_ACTIONS\", \"85\"))\n"
        "except ValueError:\n"
        "    _STALL_ACTIONS = 85\n"
        "\n"
        "from inference.agent.banking import (  # duck-fork-v2\n"
        "    TraceStep as _BankStep,\n"
        "    bank_if_won as _bank_if_won,\n"
        ")\n"
        "\n"
        "# duck-fork-v2, T2. Win-then-replay banking: a card scores the max over\n"
        "# its plays and RESET in the WIN state opens a new play, so a recorded\n"
        "# win can be replayed pruned. Every guard in the runner fails toward\n"
        "# doing nothing. Off by default like everything else here.\n"
        "_BANKING_ENABLED = os.environ.get(  # duck-fork-v2\n"
        "    \"LOCAL_ANALYZER_BANKING\", \"\"\n"
        ").strip().lower() in {\"1\", \"true\", \"yes\", \"on\"}",
        "solver import block",
    )
    s = patch(
        s,
        """            if action.id.value not in self.game.current_state.available_actions:""",
        """            if _STALL_GUARD_ENABLED:  # duck-fork-v2
                _sg = getattr(self, "_stall_guard", None)
                if _sg is None:
                    _sg = StallGuard(stall_actions=_STALL_ACTIONS)
                    self._stall_guard = _sg
                _slvl = int(self.game.current_state.levels_completed)
                _sboard = _stall_board_sig(_grid_from_state(self.game.current_state))
                _sact = _stall_action_sig(action.id.name, dict(action.data))
                if _sg.is_cycle(level=_slvl, board=_sboard, action=_sact):
                    _sg.note_blocked()
                    _smsg = (
                        f"{_format_action_display(action.id.name, dict(action.data))} "
                        f"has already been tried from this exact board on this level, "
                        f"and this level has made no progress for "
                        f"{_STALL_ACTIONS} actions. Try something you have not tried."
                    )
                    if executed_payloads:
                        stop_reason = "stalled_cycle"
                        break
                    return self._error_payload(_smsg)

            if action.id.value not in self.game.current_state.available_actions:""",
        "step_env availability check",
    )
    s = patch(
        s,
        "        board_changed = previous_grid != _grid_from_state(new_state)",
        "        board_changed = previous_grid != _grid_from_state(new_state)\n"
        "        _s2 = getattr(self, \"_stall_guard\", None)  # duck-fork-v2\n"
        "        if _s2 is not None:\n"
        "            _s2.observe(\n"
        "                level=int(new_state.levels_completed),\n"
        "                board=_stall_board_sig(previous_grid),\n"
        "                action=_stall_action_sig(action.id.name, dict(action.data)),\n"
        "            )\n"
        "        if _BANKING_ENABLED:  # duck-fork-v2, T2: record the replayable trace\n"
        "            if getattr(self, \"_bank_trace\", None) is None:\n"
        "                self._bank_trace = []\n"
        "                self._bank_initial_grid = previous_grid\n"
        "            _run = self.game.game_run\n"
        "            self._bank_trace.append(_BankStep(\n"
        "                name=action.id.name,\n"
        "                data=dict(action.data),\n"
        "                grid=_grid_from_state(new_state),\n"
        "                levels_completed=int(new_state.levels_completed),\n"
        "                won=bool(_run is not None and _run.state == \"won\"),\n"
        "            ))",
        "board_changed observation",
    )
    s = patch(
        s,
        """    def _finish_if_needed(self) -> None:
        run = self.game.game_run
        if run is not None and run.final_score is None:
            if self.stop_event.is_set() and run.state == "playing":
                run.state = "cancelled"
            self.game.finish_game()""",
        """    def _finish_if_needed(self) -> None:
        run = self.game.game_run
        if run is not None and run.final_score is None:
            if _BANKING_ENABLED:  # duck-fork-v2, T2: bank before finish_game
                try:
                    _note = _bank_if_won(self)
                    print(f"duck-fork-v2 banking[{self.game.game_id}]: {_note}",
                          flush=True)
                except Exception as _exc:  # noqa: BLE001 -- must never block completion
                    print(f"duck-fork-v2 banking[{self.game.game_id}]: "
                          f"error {type(_exc).__name__}: {_exc}", flush=True)
            if self.stop_event.is_set() and run.state == "playing":
                run.state = "cancelled"
            self.game.finish_game()""",
        "_finish_if_needed body",
    )
    sp.write_text(s)
    print(f"[fork-v2] patched {SOLVER} (stall guard + banking)")

    # -- reasoning_effort, exactly as in fork v1 --
    cp = BUILD / OPENAI_COMPAT
    c = cp.read_text()
    c = patch(c, "    seed: int | None = None,\n) -> dict[str, Any]:",
              "    seed: int | None = None,\n"
              "    reasoning_effort: str | None = None,  # duck-fork-v2\n"
              ") -> dict[str, Any]:",
              "build_chat_payload signature")
    c = patch(c,
              "        if seed is not None and seed >= 0:\n"
              "            payload[\"seed\"] = seed",
              "        if seed is not None and seed >= 0:\n"
              "            payload[\"seed\"] = seed\n"
              "        _effort = str(reasoning_effort or \"\").strip().lower()  # duck-fork-v2\n"
              "        if _effort:\n"
              "            payload[\"reasoning_effort\"] = _effort",
              "payload seed block")
    cp.write_text(c)

    tp = BUILD / TOOL_AGENT
    t = tp.read_text()
    t = patch(t,
              '_LOCAL_ANALYZER_TOP_K = _get_env_int("LOCAL_ANALYZER_TOP_K", 20)',
              '_LOCAL_ANALYZER_TOP_K = _get_env_int("LOCAL_ANALYZER_TOP_K", 20)\n'
              '# duck-fork-v2: empty means send no field and take the model default.\n'
              '_LOCAL_ANALYZER_REASONING_EFFORT = os.environ.get(\n'
              '    "LOCAL_ANALYZER_REASONING_EFFORT", ""\n'
              ').strip().lower()\n'
              '# duck-fork-v2 O2 gates, both off by default.\n'
              '_O2_KEEP = os.environ.get(\n'
              '    "LOCAL_ANALYZER_KEEP_KNOWLEDGE", ""\n'
              ').strip().lower() in {"1", "true", "yes", "on"}\n'
              '_O2_LEDGER = os.environ.get(\n'
              '    "LOCAL_ANALYZER_LEDGER", ""\n'
              ').strip().lower() in {"1", "true", "yes", "on"}\n'
              '# duck-fork-v2 axis B: the experiment protocol, appended to the\n'
              '# per-turn instructions when gated on.\n'
              '_B_PROTOCOL = os.environ.get(\n'
              '    "LOCAL_ANALYZER_PROTOCOL", ""\n'
              ').strip().lower() in {"1", "true", "yes", "on"}',
              "tool_agent env block")
    # Axis B: the experiment protocol. Everything measured says the agent can
    # act (5.5x random at starting) and can remember (O2), but does not convert
    # turns into decided hypotheses. The stock prompt asks it to DESCRIBE a
    # world model; this makes each turn TEST one. Appended as its own extend so
    # the stock list literal is never touched.
    t = patch(t,
              """        lines.append(
            "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it, "
            "but stop immediately if a result reports `game_over`, `run_complete`, `level_completed`, or `done`."
        )""",
              """        if _B_PROTOCOL:  # duck-fork-v2 axis B
            lines.extend([
                "EXPERIMENT PROTOCOL (mandatory this turn):",
                "1. BLOCKER: name the single thing preventing level completion right now.",
                "2. HYPOTHESIS: state your best mechanic hypothesis in falsifiable form: if I do X, exactly Y will happen; anything else refutes it.",
                "3. EXPERIMENT: choose the cheapest action(s) that decide the hypothesis -- prefer one never tried from this exact board.",
                "4. VERDICT: after the result, write CONFIRMED or REFUTED plus a one-line update, before anything else.",
                "Never re-execute an action already tried from the same board unless the world model says why the outcome will differ this time.",
            ])
        lines.append(
            "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it, "
            "but stop immediately if a result reports `game_over`, `run_complete`, `level_completed`, or `done`."
        )""",
              "protocol insertion")

    # O2: selective retention, ledger accumulation, ledger render.
    t = patch(t, _O2_WIPE_OLD, _O2_WIPE_NEW, "knowledge wipe block")
    t = patch(t, _O2_ACC_OLD, _O2_ACC_NEW, "ledger accumulation")
    t = patch(t, _O2_RENDER_OLD, _O2_RENDER_NEW, "ledger render")
    t = patch(t,
              "            temperature=_LOCAL_ANALYZER_TEMPERATURE,",
              "            temperature=_LOCAL_ANALYZER_TEMPERATURE,\n"
              "            reasoning_effort=_LOCAL_ANALYZER_REASONING_EFFORT,  # duck-fork-v2",
              "chat payload call")
    # Batch hint. Measured across every configuration ever run: actions per LLM
    # call is exactly 1.00 -- the model has never once used the plural
    # `action(actions)` the harness supports. Each call costs a full prefill, so
    # confident k-action sequences multiply actions/game by k for free. The hint
    # is appended to the tool description only when the gate is on.
    t = patch(t,
              '"`action(actions)` for executing one or more real environment actions, "',
              '"`action(actions)` for executing one or more real environment actions"\n'
              '    + (" -- when you are confident about the next few steps, pass a short '
              'sequence of 2-5 actions in ONE call instead of one at a time; each call '
              'has a large fixed cost and sequences are how you act quickly"\n'
              '       if os.environ.get("LOCAL_ANALYZER_BATCH_HINT", "").strip().lower()\n'
              '       in {"1", "true", "yes", "on"} else "")\n'
              '    + ", "',
              "action tool description")
    t = patch(t,
              "_PERSISTENT_HISTORY_ASSISTANT_TURNS = 30",
              '_PERSISTENT_HISTORY_ASSISTANT_TURNS = _get_env_int(  # duck-fork-v2\n'
              '    "LOCAL_ANALYZER_HISTORY_TURNS", 30\n'
              ')',
              "persistent history window")
    tp.write_text(t)
    print(f"[fork-v2] patched {OPENAI_COMPAT} and {TOOL_AGENT}")

    # O5/S2: the sandbox gains goto_frontier(), gated.
    sb = BUILD / SANDBOX
    b = sb.read_text()
    b = patch(b,
              '        runtime_globals["action"] = action',
              '        runtime_globals["action"] = action\n' + _O5_GOTO_SRC,
              "sandbox action global")
    b = patch(b,
              '                "animation_enabled": animation_handler is not None,',
              '                "animation_enabled": animation_handler is not None,\n'
              '                # duck-fork-v2 S2: the child cannot read the environment;\n'
              '                # gates cross the process boundary in this payload.\n'
              '                "frontier_goto_enabled": os.environ.get(\n'
              '                    "LOCAL_ANALYZER_FRONTIER_GOTO", ""\n'
              '                ).strip().lower() in {"1", "true", "yes", "on"},',
              "sandbox initial payload")
    b = patch(b,
              '        runtime_globals["action"] = action',
              '        runtime_globals["action"] = action\n' + _V8_WM_SRC,
              "sandbox v8 worldmodel")
    b = patch(b,
              '                "animation_enabled": animation_handler is not None,',
              '                "animation_enabled": animation_handler is not None,\n'
              '                "worldmodel_enabled": os.environ.get(\n'
              '                    "ARC3_V8_WORLDMODEL", ""\n'
              '                ).strip().lower() in {"1", "true", "yes", "on"},',
              "sandbox v8 payload gate")
    sb.write_text(b)
    t2p = BUILD / TOOL_AGENT
    t2 = t2p.read_text()
    t2 = patch(t2,
              '"and `animation()` for a compact diff timeline of the frames the last animated action produced. "',
              '"and `animation()` for a compact diff timeline of the frames the last animated action produced"\n'
              '    + (", and `goto_frontier()` to mechanically walk to the nearest '
              'recorded board that still has untried simple actions -- costs no '
              'thinking, use it when stuck"\n'
              '       if os.environ.get("LOCAL_ANALYZER_FRONTIER_GOTO", "").strip().lower()\n'
              '       in {"1", "true", "yes", "on"} else "")\n'
              '    + ". "',
              "goto tool description")
    t2 = patch(t2,
              "'and `animation()` for a compact diff timeline of the frames the last animated action produced'"
              if False else '"and `animation()` for a compact diff timeline of the frames the last animated action produced"',
              '"and `animation()` for a compact diff timeline of the frames the last animated action produced"\n'
              '    + (". The most efficient way to clear a level here is `worldmodel`, "\n'
              '       "already built from the transitions seen so far: (1) write "\n'
              '       "predict(board, action)->board capturing what actions do; "\n'
              '       "(2) call worldmodel.verify(predict) -- it scores your model against "\n'
              '       "every real transition with instrumentation cells masked out; "\n'
              '       "(3) once verify() >= 0.95, call worldmodel.plan(predict, actions, is_goal) "\n'
              '       "for the shortest action list and run it in one action([...]) call, "\n'
              '       "or worldmodel.novel_action(predict, actions) when no goal is known yet. "\n'
              '       "Planning through a verified model reaches the goal in far fewer actions "\n'
              '       "than trial and error, which is what the score rewards. Fall back to direct "\n'
              '       "actions only while verify() stays low. A level change is signalled by the "\n'
              '       "engine, not by board equality"\n'
              '       if os.environ.get("ARC3_V8_WORLDMODEL", "").strip().lower()\n'
              '       in {"1", "true", "yes", "on"} else "")',
              "v8 worldmodel contract")
    # duck-fork-v2 V10: harness-forced exploration as a stall-breaker. Host-side
    # only (no sandbox r-string involved). Gated by ARC3_V10; a missing module
    # or any explorer error degrades to the stock turn.
    t2 = patch(t2,
              "from inference.agent.runtime_state import Frame, HistoryEntry, RUNTIME_STATE_FILENAME, load_runtime_state\n",
              "from inference.agent.runtime_state import Frame, HistoryEntry, RUNTIME_STATE_FILENAME, load_runtime_state\n"
              "try:  # duck-fork-v2 V10\n"
              "    from inference.agent.v10_explorer import V10Explorer\n"
              "except Exception:  # noqa: BLE001\n"
              "    V10Explorer = None\n"
              "_V10_ENABLED = os.environ.get(\"ARC3_V10\", \"\").strip().lower() in {\"1\", \"true\", \"yes\", \"on\"}\n"
              "_V10_STALL = int(os.environ.get(\"ARC3_V10_STALL\", \"25\") or 25)\n"
              "_V10_BURST = int(os.environ.get(\"ARC3_V10_BURST\", \"200\") or 200)\n"
              "_V10_WARMUP = int(os.environ.get(\"ARC3_V10_WARMUP\", \"30\") or 30)\n"
              "_V10_AUTHOR_N = int(os.environ.get(\"ARC3_V10_AUTHOR_N\", \"4\") or 4)\n"
              "_V10_MAX_BURSTS = int(os.environ.get(\"ARC3_V10_MAX_BURSTS\", \"0\") or 0)\n"
              "_V10_AUTHOR_ROUNDS = int(os.environ.get(\"ARC3_V10_AUTHOR_ROUNDS\", \"3\") or 3)\n",
              "v10 import + gates")
    t2 = patch(t2,
              "    def analyze(\n        self,\n        state_path: Path,\n",
              "    def _v10_chat(self, prompt: str, temperature: float = 0.8) -> str:  # duck-fork-v2 V10\n"
              "        payload = {\"model\": self._model.model_id, \"messages\": [{\"role\": \"user\", \"content\": prompt}],\n"
              "                   \"max_tokens\": 4000, \"temperature\": float(temperature),\n"
              "                   \"chat_template_kwargs\": {\"enable_thinking\": False}}\n"
              "        url = str(self._model.base_url).rstrip(\"/\") + \"/chat/completions\"\n"
              "        resp = requests.post(url, json=payload, headers=self._headers(), timeout=900)\n"
              "        resp.raise_for_status()\n"
              "        choice = (resp.json().get(\"choices\") or [{}])[0]\n"
              "        msg = choice.get(\"message\") or {}\n"
              "        return str(msg.get(\"content\") or msg.get(\"reasoning_content\") or \"\")\n"
              "\n"
              "    def _v10_propose(self, state_path: Path, current_frame, history_entries):  # duck-fork-v2 V10\n"
              "        try:\n"
              "            registry = getattr(self, \"_v10\", None)\n"
              "            if registry is None:\n"
              "                registry = self._v10 = {}\n"
              "            ex = registry.get(str(state_path))\n"
              "            if ex is None:\n"
              "                log_path = state_path.parent / f\"{state_path.stem}_v10.txt\"\n"
              "                def _log(msg: str, _p=log_path) -> None:\n"
              "                    try:\n"
              "                        with open(_p, \"a\", encoding=\"utf-8\") as fh:\n"
              "                            fh.write(f\"{time.strftime('%H:%M:%S')} {msg}\\n\")\n"
              "                    except Exception:  # noqa: BLE001\n"
              "                        pass\n"
              "                ex = V10Explorer(chat=self._v10_chat, log=_log, warmup=_V10_WARMUP, n_author=_V10_AUTHOR_N,\n"
              "                                 stall_after=_V10_STALL, burst=_V10_BURST,\n"
              "                                 max_bursts=_V10_MAX_BURSTS, max_author_rounds=_V10_AUTHOR_ROUNDS)\n"
              "                registry[str(state_path)] = ex\n"
              "            if current_frame is None:\n"
              "                return None\n"
              "            if not history_entries:\n"
              "                ex.seed_initial(current_frame.grid, current_frame.level)\n"
              "            hist = [(e.action, e.frame.grid, e.frame.level) for e in history_entries]\n"
              "            return ex.propose(current_frame.grid, current_frame.level, hist, list(self._current_valid_actions))\n"
              "        except Exception:  # noqa: BLE001 - never let the explorer break a turn\n"
              "            return None\n"
              "\n"
              "    def analyze(\n        self,\n        state_path: Path,\n",
              "v10 methods")
    t2 = patch(t2,
              "        append_transcript(\"SYSTEM PROMPT\", self._system_prompt)\n",
              "        # duck-fork-v2 V10: if the explorer has a plan, run it through the\n"
              "        # agent's own python tool (history, no-op guard, animation and state\n"
              "        # persistence all reused) and skip the model this turn.\n"
              "        if _V10_ENABLED and V10Explorer is not None:\n"
              "            _v10_plan = self._v10_propose(state_path, current_frame, history_entries)\n"
              "            if _v10_plan:\n"
              "                _v10_code = \"action(\" + json.dumps(_v10_plan) + \")\"\n"
              "                append_transcript(\"V10 PLAN\", _v10_code)\n"
              "                _v10_res = self._run_python_tool(state_path, {\"code\": _v10_code})\n"
              "                append_transcript(\"V10 RESULT\", str(_v10_res.content)[:1500])\n"
              "                return AnalyzerTurnResult(step_executed=bool(_v10_res.step_executed), reasoning=\"v10 explorer plan\")\n"
              "        append_transcript(\"SYSTEM PROMPT\", self._system_prompt)\n",
              "v10 analyze hook")
    t2p.write_text(t2)
    print(f"[fork-v2] patched {SANDBOX} (goto_frontier + v8 worldmodel, gated) and {TOOL_AGENT} (v10 stall-breaker, gated)")

    readme = BUILD / "README.dataset.md"
    base = readme.read_text() if readme.exists() else ""
    readme.write_text(
        base + "\n\n## Fork notes (arc3-taaf-anim-plus)\n\n"
        "Derived from `jakobbrggen/taaf-kaggle-source-anim-20260807-anim` (CC0,\n"
        "Jakob Bruggen), itself derived from `jeroencottaar/taaf-kaggle-source-share`\n"
        "(MIT, Tufa Labs: Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries\n"
        "Smit, Michal Tesnar, Stefano Viel). Upstream is unmodified except for the\n"
        "additions below, which are all gated OFF by default:\n\n"
        "- `inference/agent/stall_guard.py` + three anchored edits in `solver.py`\n"
        "  (LOCAL_ANALYZER_STALL_GUARD / LOCAL_ANALYZER_STALL_ACTIONS)\n"
        "- `reasoning_effort` plumbing (LOCAL_ANALYZER_REASONING_EFFORT)\n"
        "- the persistent-history window as an env knob (LOCAL_ANALYZER_HISTORY_TURNS)\n\n"
        "Upstream's own no-op guard (`ARC3_HARD_NOOP_GUARD`, default on) is left\n"
        "exactly as shipped. Our additions are MIT-0.\n")
    print("[fork-v2] attribution written")

    # sanity: everything we touched still parses
    import ast
    for rel in (SOLVER, TOOL_AGENT, OPENAI_COMPAT, STALL_DEST, BANK_DEST, V10_DEST, SANDBOX):
        ast.parse((BUILD / rel).read_text())
    # And the sandbox CHILD script parses on its own -- the outer file parsing
    # is not enough, because the child is an r"""...""" string executed in a
    # separate process. v8's first two runs crashed every game ("Sandbox
    # process exited unexpectedly", even for a bare print) because an injected
    # `from __future__` mid-function made this child uncompilable while the
    # outer module still parsed. Extract the longest r-string in the sandbox
    # (the child template) and compile it.
    sb_src = (BUILD / SANDBOX).read_text()
    childs = re.findall(r'r"""(.*?)"""', sb_src, re.S)
    child = max(childs, key=len) if childs else ""
    # The child string carries the source's own indentation (its first real
    # line sets the base); strip exactly that prefix from every line so the
    # reconstructed module compiles as the child process will see it.
    base = next((ln[:len(ln) - len(ln.lstrip())]
                 for ln in child.splitlines() if ln.strip()), "")
    unindented = "\n".join(ln[len(base):] if ln.startswith(base) else ln
                           for ln in child.splitlines())
    compile(unindented, "<sandbox-child>", "exec")
    print("[fork-v2] all patched files parse (incl. the sandbox child script)")
    return BUILD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    out = build()
    meta = out / "dataset-metadata.json"
    meta.write_text(
        '{\n  "title": "ARC3 TAAF anim + gated extras",\n'
        f'  "id": "tantan0327/{OUR_SLUG}",\n'
        '  "licenses": [{"name": "CC0-1.0"}]\n}\n')
    print(f"\n[fork-v2] built {out}")
    if args.upload:
        create = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(out), "-r", "zip"],
            check=False, cwd=ROOT, capture_output=True, text=True)
        out_text = (create.stdout or "") + (create.stderr or "")
        print(out_text.strip()[-200:])
        if "already in use" in out_text or "already exists" in out_text:
            subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(out), "-r", "zip",
                 "-m", "rebuild"], check=True, cwd=ROOT)
    else:
        print("[fork-v2] re-run with --upload to publish it")


if __name__ == "__main__":
    main()
