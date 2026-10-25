# PATCH: act-rule -- turn deliberation into actions (CC0, ours), on Franzen's M2 harness.
#
# Measurement (2026-10-10, 178 game-runs): 29% of turns execute NO action (6,529 turns,
# 4,618 with an action). Their token share cannot be separated in the logs (generation is
# attributed to the next action), so it lies between 5% and ~29%. Total slot time is the
# binding constraint of the 9-hour run, so turns that produce no action are the largest
# candidate inefficiency left; this arm measures it (tokens per level AND levels). Mechanism: the harness counts consecutive turns without an executed
# action per game; once the streak reaches ARC3_ACTRULE_MAX (default 1), the next opener
# carries a rule that this turn must end in `action(...)` (a best move or the single most
# informative probe). A system line states the norm. Nothing else changes.
# Flags: ARC3_ACTRULE=0 disables; ARC3_ACTRULE_MAX=n allows n planning turns in a row.
import os as _os

from inference.agent import tool_agent as _TA

_ENABLED = _os.environ.get("ARC3_ACTRULE", "1").strip().lower() not in ("0", "false", "off", "")
_MAX = max(1, int(_os.environ.get("ARC3_ACTRULE_MAX", "1") or 1))
_STATS = {"turns": 0, "noact": 0, "forced": 0}

SYSTEM_LINE = (
    "- Every turn should end in `action(...)`. Reading the board and reasoning belong in the same "
    "turn as the move they lead to. At most one turn in a row may end without an action; after "
    "that, the turn must execute your best move or the single most informative probe.\n"
)
FORCE_LINE = (
    "Act-rule: your previous turn executed no action. Deliberation is finished for now - this turn "
    "MUST call `action(...)` with your best move, or with the one probe whose outcome you most need. "
    "Decide inside the same python call that inspects the board; do not end another turn without acting."
)

_orig_analyze = _TA.ToolAgent.analyze


def _analyze_with_actrule(self, state_path, action_num, *args, **kwargs):
    self._actrule_state_path = state_path
    return _orig_analyze(self, state_path, action_num, *args, **kwargs)


_orig_build = _TA.ToolAgent._build_user_prompt


def _build_with_actrule(self, action_num, *args, **kwargs):
    text = _orig_build(self, action_num, *args, **kwargs)
    if not _ENABLED:
        return text
    try:
        history = kwargs.get("history_entries") or []
        n = len(history)
        last_n = getattr(self, "_actrule_last_n", None)
        streak = int(getattr(self, "_actrule_streak", 0) or 0)
        if last_n is not None:
            if n <= last_n:
                streak += 1
                _STATS["noact"] += 1
            else:
                streak = 0
        self._actrule_last_n = n
        self._actrule_streak = streak
        _STATS["turns"] += 1
        if streak >= _MAX:
            _STATS["forced"] += 1
            text = text.rstrip() + "\n\n" + FORCE_LINE + "\n"
        if _STATS["turns"] % 25 == 0:
            try:
                sp = getattr(self, "_actrule_state_path", None)
                print("ACTRULE game=%s turns=%d noact=%d forced=%d" % (
                    getattr(sp, "stem", "?"), _STATS["turns"], _STATS["noact"], _STATS["forced"]), flush=True)
            except Exception:
                pass
    except Exception as exc:  # never break the turn
        try:
            print("ACTRULE hook error: %s: %s" % (type(exc).__name__, str(exc)[:160]), flush=True)
        except Exception:
            pass
    return text


_orig_system = _TA._build_system_prompt


def _system_with_actrule(*args, **kwargs):
    prompt = _orig_system(*args, **kwargs)
    if _ENABLED and SYSTEM_LINE not in prompt:
        prompt = prompt.rstrip("\n") + "\n" + SYSTEM_LINE
    return prompt


_TA.ToolAgent.analyze = _analyze_with_actrule
_TA.ToolAgent._build_user_prompt = _build_with_actrule
_TA._build_system_prompt = _system_with_actrule
print("ACTRULE_PATCH installed: enabled=%s max_planning_turns=%d" % (_ENABLED, _MAX), flush=True)
