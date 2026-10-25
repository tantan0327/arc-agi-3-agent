# PATCH: prediction-error loop (CC0, ours) on Franzen's M2 harness.
#
# Mechanism (predictive coding): before acting, the model states what it expects the action
# to change (`predict(cells=N, note="...")` inside the python call); the sandbox wrapper
# around `action(...)` measures the actual frame diff and prints a PREDICTION CHECK line into
# the same tool output: hit / miss with the real changed-cell count. Misses are the learning
# signal the model otherwise has to infer by itself. Nothing else changes.
# Flags: ARC3_PREDERR=0 disables; ARC3_PREDERR_TOL=0.5 (relative tolerance on the cell count).
import os as _os

from inference.agent import tool_agent as _TA

_ENABLED = _os.environ.get("ARC3_PREDERR", "1").strip().lower() not in ("0", "false", "off", "")
_TOL = float(_os.environ.get("ARC3_PREDERR_TOL", "0.5") or 0.5)

_PRELUDE = '''
_PRED = {"cells": None, "note": "", "hits": 0, "misses": 0, "unpredicted": 0}
def predict(cells=None, note=""):
    """State the expected effect of the NEXT action: cells = number of board cells you expect to change (0 if none), note = what moves/changes."""
    _PRED["cells"] = None if cells is None else max(0, int(cells)); _PRED["note"] = str(note)[:120]
_orig_action_for_pred = action
def action(actions):
    _before = current_frame
    _r = _orig_action_for_pred(actions)
    try:
        _d = frame_diff(_before, current_frame)
        _n = _d.get("changed_cell_count", None)
        if _n is not None and _n >= 0:
            if _PRED["cells"] is None:
                _PRED["unpredicted"] += 1
                print("PREDICTION CHECK: no prediction was stated; actual change = %d cells. State predict(cells=..., note=...) before each action." % _n)
            else:
                _e = _PRED["cells"]; _ok = abs(_n - _e) <= max(2, %TOL% * max(_e, 1))
                _PRED["hits" if _ok else "misses"] += 1
                print("PREDICTION CHECK: %s - you expected %d cells to change (%s); actual %d." % ("HIT" if _ok else "MISS", _e, _PRED["note"] or "no note", _n)
                      + ("" if _ok else " Your model of this mechanic is wrong here: revise it before the next move."))
        _PRED["cells"] = None; _PRED["note"] = ""
    except Exception:
        pass
    return _r
'''.replace("%TOL%", repr(_TOL))

RULE_LINE = (
    "Prediction rule: before every `action(...)` call, state your expectation with "
    "`predict(cells=<expected number of changed cells>, note=<what will move or change>)` in the same "
    "python call. The tool output then reports HIT or MISS against the real frame diff. A MISS means "
    "your mechanic model is wrong - fix the model before acting again; a run of HITs means you can act in batches."
)
SYSTEM_LINE = (
    "- Predict before you act: call `predict(cells=N, note=...)` before each `action(...)`; the sandbox "
    "checks it against the real change and reports HIT/MISS. Use misses to correct your understanding of the "
    "mechanics instead of trying more moves.\n"
)

_orig_run = _TA.run_sandboxed_python


def _run_with_pred(*, code, **kwargs):
    try:
        code = _PRELUDE + "\n# ---- the model's code ----\n" + str(code)
    except Exception:
        pass
    return _orig_run(code=code, **kwargs)


_orig_build = _TA.ToolAgent._build_user_prompt


def _build_with_pred(self, *args, **kwargs):
    text = _orig_build(self, *args, **kwargs)
    if _ENABLED and RULE_LINE not in text:
        text = text.rstrip() + "\n\n" + RULE_LINE + "\n"
    return text


_orig_system = _TA._build_system_prompt


def _system_with_pred(*args, **kwargs):
    prompt = _orig_system(*args, **kwargs)
    if _ENABLED and SYSTEM_LINE not in prompt:
        prompt = prompt.rstrip("\n") + "\n" + SYSTEM_LINE
    return prompt


if _ENABLED:
    _TA.run_sandboxed_python = _run_with_pred
    _TA.ToolAgent._build_user_prompt = _build_with_pred
    _TA._build_system_prompt = _system_with_pred
print("PREDERR_PATCH installed: enabled=%s tol=%.2f prelude=%d chars" % (_ENABLED, _TOL, len(_PRELUDE)), flush=True)
