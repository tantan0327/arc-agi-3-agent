# PATCH: solver consolidation + harness-verified replay, on Franzen's Milestone-2 harness.
#
# Mechanism (docs/original-algorithm.md): at every level transition the opener asks the
# model to keep ONE retained function `solve_level()`; at the start of each later level
# the harness runs that function itself, before the model's first turn, under the
# existing guards and a small action budget, and reports the trace in the opener.
# Everything else is the harness unchanged. Flag: ARC3_SOLVER_REPLAY=0 disables.
import os as _os
import json as _json
import time as _time

from inference.agent import tool_agent as _TA
from inference.agent.runtime_state import load_runtime_state as _load_state

_ENABLED = _os.environ.get("ARC3_SOLVER_REPLAY", "1").strip().lower() not in ("0", "false", "off", "")
_BUDGET_FACTOR = float(_os.environ.get("ARC3_SOLVER_REPLAY_BUDGET_FACTOR", "1.2") or 1.2)
_BUDGET_MIN = int(_os.environ.get("ARC3_SOLVER_REPLAY_BUDGET_MIN", "8") or 8)
_BUDGET_MAX = int(_os.environ.get("ARC3_SOLVER_REPLAY_BUDGET_MAX", "60") or 60)
_NOPROGRESS = int(_os.environ.get("ARC3_SOLVER_REPLAY_NOPROGRESS", "3") or 3)
_CHAIN_MAX = int(_os.environ.get("ARC3_SOLVER_REPLAY_CHAIN", "3") or 3)
_FN = "solve_level"
# Gated variant (ARC3_SOLVER_REPLAY_GATED=1): replay only a solver that has already cleared a
# level when the MODEL called it; a replay that ends in game over revokes the proof.
_GATED = _os.environ.get("ARC3_SOLVER_REPLAY_GATED", "0").strip().lower() in ("1", "true", "on")
# Prompt-only variant (ARC3_SOLVER_REPLAY_PROMPT_ONLY=1): keep the consolidation request and the system
# rule, never run the replay -- measures the prompt protocol alone.
_PROMPT_ONLY = _os.environ.get("ARC3_SOLVER_REPLAY_PROMPT_ONLY", "0").strip().lower() in ("1", "true", "on")
_STATS = {"replays": 0, "cleared": 0, "stopped": 0, "errors": 0}

CONSOLIDATE_LINE = (
    "Consolidation: keep ONE retained function `solve_level()` (no arguments) that reads "
    "`current_frame` and executes, via `action(...)`, the moves that clear a level of this "
    "game with the mechanics you have established - a general solver that handles THIS "
    "layout (locate the objects, compute the route or the clicks), not the previous level's "
    "action list. At the start of each later level the harness runs `solve_level()` itself "
    "before your first turn, with a small action budget and the usual guards, and reports what "
    "happened. Define or refine it now if you can state the rule; otherwise plan normally and "
    "define it once the mechanic is clear."
)
SYSTEM_LINE = (
    "- Once a level's mechanics are clear, define and keep `solve_level()` (no arguments): a "
    "general solver that reads `current_frame` and executes the clearing moves with `action(...)`. "
    "The harness runs it for you at the start of each later level, under the usual guards, "
    "before your first turn there.\n"
)


def _budget_for(self):
    """1.2x the actions the model spent on the level just completed, clamped."""
    try:
        spent = int(getattr(self, "_replay_prev_level_actions", 0) or 0)
    except Exception:
        spent = 0
    return max(_BUDGET_MIN, min(_BUDGET_MAX, int(round(spent * _BUDGET_FACTOR)) if spent else _BUDGET_MIN))


def _refusal(self, reason, detail):
    return {"executed": False, "executed_count": 0, "stop_reason": reason, "stop_detail": detail,
            "error": detail, "valid_actions": list(getattr(self, "_current_valid_actions", []) or [])}


def _run_replay(self, state_path, level, budget):
    """Run the retained solve_level() through the harness's own python-tool path with a
    budgeted, progress-checked step callback. Returns (trace_text, outcome)."""
    orig_cb = self._step_env_callback
    counter = {"spent": 0, "noprog": 0, "level_completed": False, "game_over": False, "last": None}

    def budgeted(step_args):
        acts = list((step_args or {}).get("actions") or [])
        remaining = budget - counter["spent"]
        if remaining <= 0:
            return _refusal(self, "replay_budget", "harness replay: action budget %d exhausted." % budget)
        if counter["noprog"] >= _NOPROGRESS:
            return _refusal(self, "replay_noprogress", "harness replay: %d actions without a board change." % counter["noprog"])
        if len(acts) > remaining:
            step_args = dict(step_args); step_args["actions"] = acts[:remaining]; acts = acts[:remaining]
        payload = orig_cb(step_args)
        if isinstance(payload, dict) and payload.get("executed"):
            n = int(payload.get("executed_count") or 1)
            counter["spent"] += n
            progressed = bool(payload.get("gameplay_changed") or payload.get("board_changed") or payload.get("level_completed"))
            counter["noprog"] = 0 if progressed else counter["noprog"] + n
            counter["level_completed"] = counter["level_completed"] or bool(payload.get("level_completed"))
            counter["game_over"] = counter["game_over"] or bool(payload.get("game_over"))
            counter["last"] = payload
        return payload

    self._step_env_callback = budgeted
    self._replay_in_progress = True
    t0 = _time.time()
    try:
        result = self._run_python_tool(state_path, {"code": "_r = %s()\nprint('solve_level returned', repr(_r)[:200])" % _FN})
        text = str(getattr(result, "content", "") or "")   # _ToolDispatchResult.content is the rendered tool payload
    except Exception as exc:  # the replay must never break a turn
        text = "error: %s: %s" % (type(exc).__name__, str(exc)[:160])
    finally:
        self._step_env_callback = orig_cb
        self._replay_in_progress = False
    if counter["game_over"]:
        self._replay_proven = False     # an unsafe solver loses its proof
    err = ""
    try:
        parsed = _json.loads(text) if text.strip().startswith("{") else None
        if isinstance(parsed, dict) and parsed.get("error"):
            err = str(parsed["error"])[:200]
    except Exception:
        pass
    if "error:" in text[:12] and not err:
        err = text[:200]
    # budget / no-progress stops take precedence over the tool payload's error text: the
    # refusal the harness returned to the model's code surfaces there as an "error"
    outcome = ("cleared" if counter["level_completed"] else "game_over" if counter["game_over"]
               else "stopped" if (counter["spent"] >= budget or counter["noprog"] >= _NOPROGRESS)
               else "error" if err else "stopped")
    summary = getattr(self, "_last_step_summary", None) or {}
    acts = summary.get("executed_actions") if isinstance(summary, dict) else None
    acts_txt = ", ".join(str(a) for a in (acts or [])[:40]) if counter["spent"] else "none"
    stop = ""
    if outcome == "stopped":
        if counter["noprog"] >= _NOPROGRESS:
            stop = "stopped after %d actions without a board change" % counter["noprog"]
        elif counter["spent"] >= budget:
            stop = "stopped at the action budget (%d)" % budget
        else:
            stop = "returned without clearing the level"
    trace = (
        "Harness replay of your retained `solve_level()` at the start of level %d (budget %d actions, %.0f s): "
        "executed %d action(s)%s. Outcome: %s%s%s. `current_frame` below is the board AFTER this replay; "
        "the actions above are already spent."
        % (level, budget, _time.time() - t0, counter["spent"], (" - " + acts_txt) if counter["spent"] else "",
           {"cleared": "the level was CLEARED by the replay", "game_over": "game over during the replay (the level reset)",
            "error": "solve_level raised an error", "stopped": stop}[outcome],
           (" - " + err) if (err and outcome == "error") else "", "")
    )
    _STATS["replays"] += 1; _STATS[{"cleared": "cleared", "stopped": "stopped", "game_over": "stopped", "error": "errors"}[outcome]] += 1
    try:
        print("SOLVER_REPLAY game=%s level=%d budget=%d executed=%d outcome=%s stats=%s" % (
            state_path.stem, level, budget, counter["spent"], outcome, _json.dumps(_STATS)), flush=True)
    except Exception:
        pass
    return trace, outcome


_orig_run_tool = _TA.ToolAgent._run_python_tool


def _run_tool_with_proof(self, state_path, arguments):
    """Record whether the model's own call of solve_level() cleared a level."""
    result = _orig_run_tool(self, state_path, arguments)
    try:
        code = str((arguments or {}).get("code", ""))
        if not getattr(self, "_replay_in_progress", False) and (_FN + "(") in code:
            summary = getattr(self, "_last_step_summary", None) or {}
            if isinstance(summary, dict) and summary.get("level_transition"):
                self._replay_proven = True
                print("SOLVER_REPLAY proof game=%s: the model's own solve_level() cleared a level" % state_path.stem, flush=True)
    except Exception:
        pass
    return result


_orig_analyze = _TA.ToolAgent.analyze


def _analyze_with_replay(self, state_path, action_num, *args, **kwargs):
    self._replay_state_path = state_path
    return _orig_analyze(self, state_path, action_num, *args, **kwargs)


_orig_build = _TA.ToolAgent._build_user_prompt


def _build_with_replay(self, action_num, *args, **kwargs):
    extra = []
    if _ENABLED:
        try:
            current_frame = kwargs.get("current_frame")
            level = int(getattr(current_frame, "level", 1) or 1) if current_frame is not None else 1
            if not hasattr(self, "_replay_levels_done"):
                self._replay_levels_done = set(); self._replay_last_level = 1; self._replay_actions_at_level_start = 0
                self._replay_prev_level_actions = 0
            history = kwargs.get("history_entries") or []
            if level != self._replay_last_level:
                # a new level: remember how many actions the previous one cost
                self._replay_prev_level_actions = max(0, len(history) - int(self._replay_actions_at_level_start or 0))
                self._replay_actions_at_level_start = len(history)
                self._replay_last_level = level
            state_path = getattr(self, "_replay_state_path", None)
            chain = 0
            while (not _PROMPT_ONLY and level >= 2 and level not in self._replay_levels_done and state_path is not None
                   and _FN in (getattr(self, "_kept_functions", None) or {}) and chain < _CHAIN_MAX
                   and (not _GATED or getattr(self, "_replay_proven", False))):
                self._replay_levels_done.add(level); chain += 1
                trace, outcome = _run_replay(self, state_path, level, _budget_for(self))
                extra.append(trace)
                # refresh what the opener will describe
                current_frame, history = _load_state(state_path)
                kwargs["current_frame"] = current_frame; kwargs["history_entries"] = history
                kwargs["previous_step_summary"] = getattr(self, "_last_step_summary", None)
                new_level = int(getattr(current_frame, "level", level) or level)
                if new_level == level:
                    break
                self._replay_prev_level_actions = max(0, len(history) - int(self._replay_actions_at_level_start or 0))
                self._replay_actions_at_level_start = len(history); self._replay_last_level = new_level; level = new_level
            if level >= 2 and extra == [] and level not in getattr(self, "_replay_nudged", set()):
                pass
        except Exception as exc:  # never break the turn
            try:
                print("SOLVER_REPLAY hook error: %s: %s" % (type(exc).__name__, str(exc)[:160]), flush=True)
            except Exception:
                pass
    text = _orig_build(self, action_num, *args, **kwargs)
    if _ENABLED:
        try:
            # the consolidation request rides on every level-transition opener
            if "Level completed" in text or "You have completed the previous level" in text or extra:
                lines = [text.rstrip()] + extra + [CONSOLIDATE_LINE]
                text = "\n\n".join(lines) + "\n"
        except Exception:
            pass
    return text


_orig_system = _TA._build_system_prompt


def _system_with_replay(*args, **kwargs):
    prompt = _orig_system(*args, **kwargs)
    if _ENABLED and SYSTEM_LINE not in prompt:
        prompt = prompt.rstrip("\n") + "\n" + SYSTEM_LINE
    return prompt


_TA.ToolAgent._run_python_tool = _run_tool_with_proof
_TA.ToolAgent.analyze = _analyze_with_replay
_TA.ToolAgent._build_user_prompt = _build_with_replay
_TA._build_system_prompt = _system_with_replay
print("SOLVER_REPLAY_PATCH installed: enabled=%s gated=%s prompt_only=%s budget x%.1f [%d,%d] noprogress=%d chain=%d" % (
    _ENABLED, _GATED, _PROMPT_ONLY, _BUDGET_FACTOR, _BUDGET_MIN, _BUDGET_MAX, _NOPROGRESS, _CHAIN_MAX), flush=True)
