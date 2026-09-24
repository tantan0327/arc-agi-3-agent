"""Generate the notebook patch that gives the model the goto tools.

Two hooks, both on names the agent module already binds:

1. `tool_agent.run_sandboxed_python` is wrapped so the prelude
   (`duck_delta/goto_tool.py`, stdlib-only) is prepended to the model's code on
   every python-tool call. The helpers `body()`, `goto(row, col)` and
   `goto_colour(colour)` then exist inside the sandbox next to `action()`.
2. `ToolAgent._build_user_prompt` (the method that actually renders the user
   prompt -- `_describe_last_outcome` is dead code, learned the hard way) gets
   one short paragraph telling the model the helpers exist and when to use them.

Also raises the python-tool timeout to 90 s (the 9/19 facts run showed the
30 s default was costing turns: timeouts 5 -> 1).

    python scripts/build_goto_patch.py > scratchpad/goto_patch.py
    python scripts/build_probe.py gotofull --patch scratchpad/goto_patch.py --seconds 7920
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "duck_delta" / "goto_tool.py"

HOOK = '''
import os as _os

_os.environ["LOCAL_ANALYZER_TOOL_TIMEOUT"] = _os.environ.get("ARC3_TOOL_TIMEOUT", "90")

from inference.agent import tool_agent as _TA

_orig_run = _TA.run_sandboxed_python


def _run_with_goto(*, code, timeout_seconds, initial_state, action_handler):
    try:
        code = _GOTO_PRELUDE + "\\n\\n# ---- the model's code ----\\n" + str(code)
        _TA._GOTO_CALLS = getattr(_TA, "_GOTO_CALLS", 0) + 1
    except Exception:
        pass
    return _orig_run(code=code, timeout_seconds=timeout_seconds, initial_state=initial_state, action_handler=action_handler)


_TA.run_sandboxed_python = _run_with_goto

_HELPER_TEXT = (
    "Extra helpers inside `python`, next to `action`: `body()` returns what the arrow keys move "
    "(its colour, top-left, size, step per key, and the colours that blocked it), measured from your own "
    "recorded moves. `goto(row, col)` plans a route with those measured rules to put the body's top-left "
    "at (row, col), executes it, re-plans if the board disagrees, and returns a dict with `reached`, "
    "`body_at`, `actions_taken` and a per-step `trace`. `goto_colour(c)` does the same until the body "
    "touches a cell of colour c (c is the colour INDEX 0-15, not the letter). Prefer these over typing long "
    "arrow sequences by hand once a few arrow moves have been recorded; they need at least four moves that "
    "changed the board to learn the rules. They only handle arrow keys."
)

_orig_build = _TA.ToolAgent._build_user_prompt


def _build_with_helpers(self, *args, **kwargs):
    text = _orig_build(self, *args, **kwargs)
    try:
        if _HELPER_TEXT not in text:
            text = text.rstrip() + "\\n\\n" + _HELPER_TEXT + "\\n"
            _TA._GOTO_SHOWN = getattr(_TA, "_GOTO_SHOWN", 0) + 1
    except Exception:
        pass
    return text


_TA.ToolAgent._build_user_prompt = _build_with_helpers
print("GOTO_PATCH installed: prelude %d chars, tool timeout %s s" % (len(_GOTO_PRELUDE), _os.environ["LOCAL_ANALYZER_TOOL_TIMEOUT"]), flush=True)
'''


def main() -> None:
    src = TOOL.read_text()
    print("# PROBE PATCH: goto tools -- the physics planner as helpers the model calls")
    print("_GOTO_PRELUDE = " + repr(src))
    print(HOOK)


if __name__ == "__main__":
    main()
