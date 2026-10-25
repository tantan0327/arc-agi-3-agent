# PATCH: orienting census at level start, on Franzen's Milestone-2 harness.
#
# Mechanism: before the model's first turn on a level, the harness itself executes each
# simple valid action once (UP, DOWN, LEFT, RIGHT, SPACE - never MOUSE/UNDO/RESET), in a
# fixed order, through the harness's own python-tool path (guards, animation, state refresh
# all unchanged), stops at any terminal event, and reports a compact per-action effects
# table in the opener. No reversal: UNDO is advertised by the engine but reverts the board
# in 1 of 25 local games (scripts/test_undo_offline.py, 2026-10-06), so the model's first
# turn starts from the post-census board, which the opener says plainly.
# Flags: ARC3_CENSUS=0 disables; ARC3_CENSUS_LEVELS=all|first (default all);
#        ARC3_CENSUS_ACTIONS=UP,DOWN,LEFT,RIGHT,SPACE; ARC3_CENSUS_MAX=5.
import os as _os
import json as _json
import time as _time

from inference.agent import tool_agent as _TA
from inference.agent.runtime_state import load_runtime_state as _load_state

_ENABLED = _os.environ.get("ARC3_CENSUS", "1").strip().lower() not in ("0", "false", "off", "")
_LEVELS = _os.environ.get("ARC3_CENSUS_LEVELS", "all").strip().lower()
_ACTIONS = [a.strip().upper() for a in _os.environ.get("ARC3_CENSUS_ACTIONS", "UP,DOWN,LEFT,RIGHT,SPACE").split(",") if a.strip()]
_MAX = int(_os.environ.get("ARC3_CENSUS_MAX", "5") or 5)
_DIFF_TOKENS = int(_os.environ.get("ARC3_CENSUS_DIFF_TOKENS", "110") or 110)
_STATS = {"censuses": 0, "probes": 0, "executed": 0, "cleared": 0, "game_over": 0, "errors": 0}

SYSTEM_LINE = (
    "- Orienting census: at the start of each level the harness itself executes each simple "
    "valid action once (UP, DOWN, LEFT, RIGHT, SPACE; never MOUSE), in that order, before your "
    "first turn there, and reports what each one changed. Those actions are already spent and "
    "NOT undone: `current_frame` is the board after them. Use the table as your first "
    "mechanics evidence instead of repeating the probes.\n"
)

_PROBE_CODE = (
    "import json as _json\n"
    "_b = current_frame\n"
    "_r = action(%r)\n"
    "_d = frame_diff(_b, current_frame)\n"
    "_keys = ('executed', 'executed_actions', 'gameplay_changed', 'board_changed', 'level_completed', "
    "'game_over', 'run_complete', 'stop_reason', 'stop_detail', 'level')\n"
    "print('CENSUS_JSON ' + _json.dumps({'diff': _d, 'result': {k: _r.get(k) for k in _keys if k in _r}}))\n"
)


def _parse_tool_text(text):
    """-> (stdout, error) from the rendered tool payload (a JSON object string)."""
    try:
        payload = _json.loads(text) if text.strip().startswith("{") else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return str(payload.get("stdout") or ""), str(payload.get("error") or "")


def _probe(self, state_path, name):
    """Execute one census action through the harness python tool. Returns a record."""
    rec = {"action": name, "executed": False, "outcome": "", "lines": []}
    try:
        result = self._run_python_tool(state_path, {"code": _PROBE_CODE % name})
        text = str(getattr(result, "content", "") or "")
    except Exception as exc:
        rec["outcome"] = "harness error: %s" % type(exc).__name__
        _STATS["errors"] += 1
        return rec
    stdout, error = _parse_tool_text(text)
    data = None
    for line in stdout.splitlines():
        if line.startswith("CENSUS_JSON "):
            try:
                data = _json.loads(line[len("CENSUS_JSON "):])
            except Exception:
                data = None
    if data is None:
        err = (error or stdout or text).strip().replace("\n", " ")
        low = err.lower()
        if "no change" in low or "known_noop" in low or "nooperror" in low:
            rec["outcome"] = "not executed: known no-op from this board (no action spent)"
        elif "fatal" in low or "death" in low or "ended a previous attempt" in low:
            rec["outcome"] = "not executed: known fatal from this board (no action spent)"
        else:
            rec["outcome"] = "not executed: " + err[:140]
            _STATS["errors"] += 1
        return rec
    res = data.get("result") or {}
    diff = data.get("diff") or {}
    rec["executed"] = bool(res.get("executed"))
    rec["level_completed"] = bool(res.get("level_completed"))
    rec["game_over"] = bool(res.get("game_over"))
    rec["run_complete"] = bool(res.get("run_complete"))
    if not rec["executed"]:
        rec["outcome"] = "not executed: %s" % str(res.get("stop_detail") or res.get("stop_reason") or "refused")[:140]
        return rec
    _STATS["executed"] += 1
    if rec["run_complete"]:
        rec["outcome"] = "the RUN was completed by this action"
    elif rec["level_completed"]:
        rec["outcome"] = "LEVEL CLEARED by this action"
        _STATS["cleared"] += 1
    elif rec["game_over"]:
        rec["outcome"] = "GAME OVER - this action ended the attempt (the level was reset)"
        _STATS["game_over"] += 1
    else:
        changed = diff.get("changed_cell_count")
        if res.get("gameplay_changed") is False and (changed in (0, None) or not res.get("board_changed")):
            rec["outcome"] = "no change"
        elif res.get("gameplay_changed") is False:
            rec["outcome"] = "no change inside the board area (only the border/bar moved)"
        else:
            try:
                rows, cols = None, None
                frame, _ = _load_state(state_path)
                shape = getattr(frame, "shape", None)
                if shape:
                    rows, cols = int(shape[0]), int(shape[1])
                lines = _TA._render_auto_frame_diff(diff, _DIFF_TOKENS, (rows * cols) if rows and cols else None)
            except Exception:
                lines = ["%s cells changed" % changed]
            rec["lines"] = [str(l).strip() for l in lines if str(l).strip()]
            rec["outcome"] = " ".join(rec["lines"]) if rec["lines"] else "%s cells changed" % changed
    return rec


def _run_census(self, state_path, level):
    valid = [str(a).strip().upper() for a in (getattr(self, "_current_valid_actions", None) or [])]
    names = [a for a in _ACTIONS if a in valid][:_MAX]
    t0 = _time.time()
    records = []
    stopped = ""
    for name in names:
        rec = _probe(self, state_path, name)
        records.append(rec)
        _STATS["probes"] += 1
        if rec.get("level_completed") or rec.get("game_over") or rec.get("run_complete"):
            stopped = name
            break
        if rec["outcome"].startswith("harness error"):
            break
    _STATS["censuses"] += 1
    n_exec = sum(1 for r in records if r["executed"])
    skipped = [n for n in names if n not in [r["action"] for r in records]]
    head = (
        "Orienting census (done by the harness at the start of level %d, before this turn): each simple valid "
        "action was executed once from the level's starting board, in this order. These %d action(s) are already "
        "spent and were NOT undone - `current_frame` below is the board AFTER them."
        % (level, n_exec)
    )
    body = []
    for r in records:
        body.append("- %s: %s" % (r["action"], r["outcome"]))
    if stopped:
        body.append("(census stopped at %s%s)" % (stopped, ("; not probed: " + ", ".join(skipped)) if skipped else ""))
    if not names:
        head = ("Orienting census: no simple actions (UP/DOWN/LEFT/RIGHT/SPACE) are valid in this game, so the harness "
                "executed nothing at the start of level %d." % level)
    try:
        print("CENSUS game=%s level=%d probes=%d executed=%d stopped=%s %.1fs stats=%s" % (
            state_path.stem, level, len(records), n_exec, stopped or "-", _time.time() - t0, _json.dumps(_STATS)), flush=True)
    except Exception:
        pass
    return "\n".join([head] + body), records


_orig_analyze = _TA.ToolAgent.analyze


def _analyze_with_census(self, state_path, action_num, *args, **kwargs):
    self._census_state_path = state_path
    return _orig_analyze(self, state_path, action_num, *args, **kwargs)


_orig_build = _TA.ToolAgent._build_user_prompt


def _build_with_census(self, action_num, *args, **kwargs):
    extra = []
    if _ENABLED:
        try:
            current_frame = kwargs.get("current_frame")
            level = int(getattr(current_frame, "level", 1) or 1) if current_frame is not None else 1
            if not hasattr(self, "_census_levels_done"):
                self._census_levels_done = set()
            state_path = getattr(self, "_census_state_path", None)
            pre_summary = kwargs.get("previous_step_summary")
            wanted = (_LEVELS != "first" or level <= 1) and level not in self._census_levels_done
            if wanted and state_path is not None and getattr(self, "_step_env_callback", None) is not None \
                    and not (isinstance(pre_summary, dict) and pre_summary.get("run_complete")):
                self._census_levels_done.add(level)
                text, records = _run_census(self, state_path, level)
                extra.append(text)
                # the opener must describe the board AFTER the census
                current_frame, history = _load_state(state_path)
                kwargs["current_frame"] = current_frame
                kwargs["history_entries"] = history
                post = getattr(self, "_last_step_summary", None)
                merged = dict(post) if isinstance(post, dict) else (dict(pre_summary) if isinstance(pre_summary, dict) else None)
                if isinstance(merged, dict) and isinstance(pre_summary, dict) and pre_summary.get("level_transition") \
                        and not merged.get("game_over") and not merged.get("level_transition"):
                    merged["level_transition"] = True   # keep the level-start guidance the base harness gives
                if isinstance(merged, dict):
                    kwargs["previous_step_summary"] = merged
                new_level = int(getattr(current_frame, "level", level) or level)
                if new_level != level:
                    # a probe cleared the level: census the next one too (once)
                    self._census_levels_done.add(new_level)
                    text2, _ = _run_census(self, state_path, new_level)
                    extra.append(text2)
                    current_frame, history = _load_state(state_path)
                    kwargs["current_frame"] = current_frame
                    kwargs["history_entries"] = history
                    post = getattr(self, "_last_step_summary", None)
                    if isinstance(post, dict):
                        m2 = dict(post); m2["level_transition"] = True if not m2.get("game_over") else m2.get("level_transition")
                        kwargs["previous_step_summary"] = m2
        except Exception as exc:  # never break the turn
            try:
                print("CENSUS hook error: %s: %s" % (type(exc).__name__, str(exc)[:160]), flush=True)
            except Exception:
                pass
    text = _orig_build(self, action_num, *args, **kwargs)
    if extra:
        text = "\n\n".join([text.rstrip()] + extra) + "\n"
    return text


_orig_system = _TA._build_system_prompt


def _system_with_census(*args, **kwargs):
    prompt = _orig_system(*args, **kwargs)
    if _ENABLED and SYSTEM_LINE not in prompt:
        prompt = prompt.rstrip("\n") + "\n" + SYSTEM_LINE
    return prompt


_TA.ToolAgent.analyze = _analyze_with_census
_TA.ToolAgent._build_user_prompt = _build_with_census
_TA._build_system_prompt = _system_with_census
print("CENSUS_PATCH installed: enabled=%s levels=%s actions=%s max=%d" % (_ENABLED, _LEVELS, ",".join(_ACTIONS), _MAX), flush=True)
