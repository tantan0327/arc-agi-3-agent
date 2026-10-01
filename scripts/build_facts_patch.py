"""Generate the notebook patch that turns the physics planner into a perception
oracle: it never acts, it REPORTS.

Diagnosis (2026-09-16): on g50t, ls20 and m0r0 the agent had the mechanic
substantially right and lost its level-2 turns to its own perception code --
confusing the player with the goal, mis-estimating the cell pitch, walking
into walls, re-parsing every history frame until the 30-second tool timeout.
`duck_delta/physics_planner.py` learns exactly those facts from the same
transitions: which body the keys move, its position, the step per key, which
colours blocked it, which colour reset the level. This patch appends one
"Mechanical facts" block to the next user prompt whenever the fit is credible,
and raises the tool timeout so a slow parse costs seconds, not a turn.

    python scripts/build_facts_patch.py > scratchpad/facts_patch.py
    python scripts/build_probe.py factsfull --patch scratchpad/facts_patch.py --seconds 7920
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "duck_delta" / "physics_planner.py"

HOOK = '''

# ---------------------------------------------------------------- the oracle hook
import os as _os
import time as _time

_os.environ["LOCAL_ANALYZER_TOOL_TIMEOUT"] = _os.environ.get("ARC3_TOOL_TIMEOUT", "90")

from inference.agent import tool_agent as _TA
from inference.agent.runtime_state import load_runtime_state as _load_state

_orig_analyze = _TA.ToolAgent.analyze
_MIN_MOVES = int(_os.environ.get("ARC3_FACTS_MIN_MOVES", "8") or 8)


class _Oracle:
    """Ingest the game's history, fit the physics once it is credible, and
    render the facts. Never proposes an action."""

    def __init__(self, log):
        self.stage = PhysicsStage(log=log, stall_after=10**9, min_moves=_MIN_MOVES)
        self.last_text = ""

    def facts(self, grid, level, history):
        st = self.stage
        st._ingest(history)
        if st.phys is None and not st._fit(level):
            return ""
        P = st.phys
        if P is None or not P.role_tables:
            return ""
        g = np.asarray(grid, dtype=np.int16)
        bodies = entities_of(g, P.mover, P.chrome, shapes=P.shapes)
        lines = ["Mechanical facts measured from your own moves so far (trust these over guesses):"]
        if bodies:
            pos = ", ".join(f"rows {t}-{t + max(o[0] for o in offs)} cols {l}-{l + max(o[1] for o in offs)}"
                            for t, l, offs, _ in bodies[:4])
            lines.append(f"- The body the arrow keys move is colour {P.mover} ({len(bodies)} body/bodies), now at {pos}.")
        for k, tab in enumerate(P.role_tables[:4]):
            if tab:
                moves = ", ".join(f"{a}->({u[0]:+d},{u[1]:+d})" for a, u in sorted(tab.items()))
                lines.append(f"- Body {k}: unit direction per key {moves}; step size per key {P.role_steps or P.role_step} cells.")
        if P.floor is not None:
            lines.append(f"- Floor colour (walkable): {P.floor}; other colours seen under successful moves: {sorted(c for c in P.passable if c != P.floor)}.")
        if P.blocking:
            lines.append(f"- Colours that BLOCKED a move: {sorted(P.blocking)}.")
        if P.deadly:
            lines.append(f"- Colours that RESET the level on entry: {sorted(P.deadly)}.")
        if P.goal_kind in ("collide", "reach"):
            what = "the two bodies touching or crossing" if P.goal_kind == "collide" else f"the body reaching colour {sorted(P.goal_colours)}"
            lines.append(f"- A previous level was completed by {what}.")
        if len(lines) == 1:
            return ""
        return "\\n".join(lines)


def _oracle_for(self, state_path):
    reg = getattr(self, "_facts", None)
    if reg is None:
        reg = self._facts = {}
    o = reg.get(str(state_path))
    if o is None:
        log_path = state_path.parent / (state_path.stem + "_facts.txt")

        def _log(msg, _p=log_path):
            try:
                with open(_p, "a", encoding="utf-8") as fh:
                    fh.write(_time.strftime("%H:%M:%S ") + str(msg) + "\\n")
            except Exception:
                pass

        o = reg[str(state_path)] = _Oracle(_log)
    return o


def _analyze_with_facts(self, state_path, action_num, *args, **kwargs):
    try:
        current_frame, history_entries = _load_state(state_path)
        if current_frame is not None:
            o = _oracle_for(self, state_path)
            if not history_entries:
                o.stage.seed_initial(current_frame.grid, current_frame.level)
            hist = [(e.action, e.frame.grid, e.frame.level) for e in history_entries]
            o.last_text = o.facts(current_frame.grid, current_frame.level, hist)
            self._facts_text = o.last_text
            if o.last_text:
                _TA._FACTS_SHOWN = getattr(_TA, "_FACTS_SHOWN", 0) + 1
    except Exception as exc:  # never let the oracle break a turn
        try:
            print("FACTS_HOOK error " + type(exc).__name__ + ": " + str(exc)[:160], flush=True)
        except Exception:
            pass
    return _orig_analyze(self, state_path, action_num, *args, **kwargs)


_orig_build = _TA.ToolAgent._build_user_prompt


def _build_with_facts(self, *args, **kwargs):
    text = _orig_build(self, *args, **kwargs)
    extra = getattr(self, "_facts_text", "")
    if extra and extra not in text:
        text = text.rstrip() + "\\n\\n" + extra + "\\n"
        _TA._FACTS_SHOWN = getattr(_TA, "_FACTS_SHOWN", 0) + 1
    return text


_TA.ToolAgent.analyze = _analyze_with_facts
_TA.ToolAgent._build_user_prompt = _build_with_facts
print("FACTS_HOOK installed (min_moves=%d, tool timeout=%s s)" % (_MIN_MOVES, _os.environ["LOCAL_ANALYZER_TOOL_TIMEOUT"]), flush=True)
'''


def main() -> None:
    source = MODULE.read_text().replace("from __future__ import annotations\n", "")
    print("# PROBE PATCH: the physics planner as a perception oracle (facts in the prompt, no actions)")
    print(source)
    print(HOOK)


if __name__ == "__main__":
    main()
