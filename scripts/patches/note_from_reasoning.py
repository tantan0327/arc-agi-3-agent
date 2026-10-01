# PROBE PATCH: keep the world-model note alive when the model writes no prose.
#
# Diagnosis (2026-09-16, transcripts of six full runs): the harness carries a
# "Working world model" note across turns, but it is updated only from the
# ASSISTANT'S PROSE (`_update_summarized_knowledge_from_assistant(content)`,
# called when a turn ends with text and no tool call). This model thinks and
# then calls the tool directly, so `content` is empty on most turns -- the
# note was empty on 55/55 turns of one bp35 run, 47/47 of another, 53/53 of a
# sk48 run. Cross-turn memory silently depends on the model's verbosity, and
# on those games the agent re-derives the game every turn from scratch.
#
# The same labelled lines (`World model:`, `Goal model:`, `Plan:` ...) that the
# system prompt asks for do appear inside the THINKING. So when the note is
# empty at prompt-building time, backfill it from the most recent assistant
# message's retained reasoning, with the harness's own parser.
from inference.agent import tool_agent as _TA

_orig_lines = _TA.ToolAgent._summarized_knowledge_lines
_KEYS = ("world_model", "goal_model", "action_model", "recent_findings", "open_questions", "current_plan")


def _backfill_from_reasoning(self):
    try:
        know = getattr(self, "_summarized_knowledge", None)
        if know is None or any(know.get(k) for k in _KEYS):
            return
        for message in reversed(getattr(self, "_history_messages", []) or []):
            if str(message.get("role", "")).strip() != "assistant":
                continue
            reasoning = _TA._extract_reasoning_text(message)
            if not reasoning:
                continue
            note = _TA._extract_scientist_note(reasoning)
            if note and any(note.get(k) for k in _KEYS):
                for key, value in note.items():
                    if value:
                        know[key] = value
                _TA._NOTE_BACKFILLS = getattr(_TA, "_NOTE_BACKFILLS", 0) + 1
            break                       # only the latest assistant turn is current
    except Exception as exc:  # never let the probe break a turn
        try:
            print("NOTE_PATCH error " + type(exc).__name__ + ": " + str(exc)[:120], flush=True)
        except Exception:
            pass


def _lines_with_backfill(self):
    _backfill_from_reasoning(self)
    return _orig_lines(self)


_TA.ToolAgent._summarized_knowledge_lines = _lines_with_backfill
print("NOTE_PATCH installed: world-model note backfilled from retained reasoning when prose is empty", flush=True)
