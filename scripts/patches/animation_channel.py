# PROBE PATCH: give the model the frames the harness throws away.
#
# Diagnosis (2026-09-16, from the transcripts of six full runs read against
# the human replays): the framework's Game exposes only the LAST frame of each
# action (`taaf/game.py`: `Frame(data=self.raw.frame[-1])`); the intermediate
# frames are available as `state.animation_frames` and nobody reads them. In
# sb26 the level's only feedback -- outputs filling one by one, then the red
# flash at the mismatching slot -- lives in those frames, and the agent
# submitted eleven arrangements, three of them already falsified, because
# every submission looked like a one-cell change to the energy bar. In sp80
# the liquid pour is entirely animated: 185 actions, 36 SPACE, seven deaths,
# and the agent never saw liquid once. Our earlier fork's notes: 49 of 49
# surviving prediction contradictions were distinguishable only in the
# animation frames.
#
# This patch adds one sentence per animated action, in two places the model
# already reads: `last_action_result["animation_text"]` inside the python tool,
# and the "The code executed N actions..." line of the next user prompt. Single
# frame actions carry no extra tokens at all. Ported from the fork's
# `inference/utils/animation.py` (summarize_animation / describe_animation).
from typing import Any, Sequence

from inference.framework import solver as _S
from inference.agent import tool_agent as _TA


def _norm(raw_frames: Any) -> list:
    frames = []
    for frame in raw_frames or ():
        rows = frame.tolist() if hasattr(frame, "tolist") else frame
        frames.append(tuple(tuple(int(c) for c in row) for row in rows or ()))
    return frames


def _transient(frames: Sequence) -> list:
    final = frames[-1]
    cells = set()
    for frame in frames[:-1]:
        for r, row in enumerate(frame):
            frow = final[r] if r < len(final) else ()
            for c, v in enumerate(row):
                fv = frow[c] if c < len(frow) else None
                if v != fv:
                    cells.add((r, c))
    return sorted(cells)


def summarize_animation(frames: Sequence, *, board_changed: bool):
    if len(frames) <= 1:
        return None
    tr = _transient(frames)
    s = {"frames": len(frames), "unique_frames": len(dict.fromkeys(frames)),
         "board_unchanged": not board_changed, "transient_pixels": len(tr)}
    if tr:
        rows = [c[0] for c in tr]; cols = [c[1] for c in tr]
        s["transient_bbox"] = [min(rows), min(cols), max(rows), max(cols)]
        # where the transient cells sit, as a coarse 8x8 map of 8-cell blocks,
        # so the model can localise the effect without a coordinate list
        blocks = sorted({(r // 8, c // 8) for r, c in tr})
        s["transient_blocks"] = blocks[:12]
    return s


def describe_animation(s) -> str:
    if not s:
        return ""
    pieces = [f"That action animated over {s['frames']} frames ({s['unique_frames']} distinct)"]
    if s.get("board_unchanged"):
        pieces.append("and the final board equals the one before it -- the effect is visible ONLY in the intermediate frames")
    tp = s.get("transient_pixels") or 0
    if tp:
        b = s.get("transient_bbox")
        region = f" around rows {b[0]}-{b[2]}, cols {b[1]}-{b[3]}" if b else ""
        blocks = s.get("transient_blocks")
        grid = (" (8x8-block cells " + ", ".join(f"({r},{c})" for r, c in blocks) + ")") if blocks else ""
        pieces.append(f"with {tp} transient cells{region}{grid}")
    else:
        pieces.append("with no transient cells (pure motion)")
    return ", ".join(pieces) + "."


# ---- solver side: attach the summary to every action payload ---------------
# the per-game session object owns _execute_action (solver.py:684); fall back
# to a scan only if the bundle renamed it
_target_cls = getattr(_S, "_HarnessGameSession", None)
if _target_cls is None or "_execute_action" not in vars(_target_cls):
    _target_cls = next((getattr(_S, n) for n in dir(_S)
                        if isinstance(getattr(_S, n), type) and "_execute_action" in vars(getattr(_S, n))), None)
if _target_cls is None:
    raise RuntimeError("animation patch: no class defines _execute_action in inference.framework.solver")
_orig_execute = _target_cls._execute_action


def _execute_with_animation(self, action, **kwargs):
    payload = _orig_execute(self, action, **kwargs)
    try:
        state = self.game.current_state
        frames = _norm(getattr(state.raw, "frame", None))
        summary = summarize_animation(frames, board_changed=bool(payload.get("board_changed")))
        if summary:
            payload["animation"] = summary
            payload["animation_text"] = describe_animation(summary)
            _S._ANIM_COUNT = getattr(_S, "_ANIM_COUNT", 0) + 1
            try:
                self._anim_seen.append(summary)
            except Exception:
                self._anim_seen = [summary]
    except Exception as exc:  # never let the channel break an action
        try:
            print("ANIMATION_PATCH error " + type(exc).__name__ + ": " + str(exc)[:120], flush=True)
        except Exception:
            pass
    return payload


_target_cls._execute_action = _execute_with_animation

# ---- batches: the harness returns the LAST action's payload for a batch, so an
# animation earlier in the batch would vanish; keep the richest one seen
# in this bundle the batch loop is `_HarnessGameSession.step_env` (solver.py:605)
_batch_name = next((n for n in ("step_env", "_execute_action_batch", "_execute_actions")
                    if n in vars(_target_cls)), None)
if _batch_name is None:
    _batch_name = next((n for n, f in vars(_target_cls).items()
                        if callable(f) and n != "_execute_action"
                        and "executed_payloads" in getattr(getattr(f, "__code__", None), "co_varnames", ())), None)
if _batch_name:
    _orig_batch = getattr(_target_cls, _batch_name)

    def _batch_with_animation(self, *args, **kwargs):
        self._anim_seen = []
        payload = _orig_batch(self, *args, **kwargs)
        try:
            seen = [a for a in getattr(self, "_anim_seen", []) if a]
            if isinstance(payload, dict) and seen:
                best = max(seen, key=lambda a: (int(a.get("transient_pixels") or 0), int(a.get("frames") or 0)))
                payload["animation"] = best
                payload["animation_text"] = describe_animation(best)
        except Exception:
            pass
        return payload

    setattr(_target_cls, _batch_name, _batch_with_animation)


# ---- agent side: the harness whitelists action-result keys (tool_agent.py
# `_compact_action_result`), which is where the first run lost the sentence;
# carry the two keys through, then into the step summary and the prompt ----
_orig_compact = _TA.ToolAgent._compact_action_result


def _compact_with_animation(self, payload):
    compact = _orig_compact(self, payload)
    try:
        if isinstance(payload, dict):
            for key in ("animation", "animation_text"):
                if payload.get(key):
                    compact[key] = payload[key]
    except Exception:
        pass
    return compact


_TA.ToolAgent._compact_action_result = _compact_with_animation

_orig_summarize = _TA.ToolAgent._summarize_step_sequence


def _summarize_with_animation(self, action_results):
    summary = _orig_summarize(self, action_results)
    try:
        if summary is not None:
            texts = []
            for item in action_results or ():
                if not isinstance(item, dict):
                    continue
                a = item.get("animation")
                if isinstance(a, dict) and a:
                    texts.append((int(a.get("transient_pixels") or 0), int(a.get("frames") or 0), item.get("animation_text") or describe_animation(a)))
            if texts:
                texts.sort(reverse=True)
                summary["animation_text"] = texts[0][2]
    except Exception:
        pass
    return summary


# `_describe_last_outcome` exists but is never called; the previous step is
# rendered inline by `_build_user_prompt`, so that is where the sentence goes
_orig_build = _TA.ToolAgent._build_user_prompt


def _build_with_animation(self, *args, **kwargs):
    text = _orig_build(self, *args, **kwargs)
    try:
        summary = kwargs.get("previous_step_summary")
        if summary is None and len(args) >= 5:
            summary = args[4]
        if summary is None:
            summary = getattr(self, "_last_step_summary", None)
        extra = (summary or {}).get("animation_text")
        if extra and extra not in text:
            text = text.rstrip() + "\n" + extra + "\n"
            _TA._ANIM_SHOWN = getattr(_TA, "_ANIM_SHOWN", 0) + 1
    except Exception:
        pass
    return text


_TA.ToolAgent._summarize_step_sequence = _summarize_with_animation
_TA.ToolAgent._build_user_prompt = _build_with_animation
print("ANIMATION_PATCH installed on %s._execute_action and ToolAgent step summary" % _target_cls.__name__, flush=True)
