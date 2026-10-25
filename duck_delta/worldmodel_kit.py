"""The equality a world-model verifier can trust (v8-D1).

Rodionov-style executable world models verify `predict(board, action)` by
exact frame comparison and stop on divergence. Our §5 measurement says exact
comparison is wrong 57% of the time on this benchmark's rendering: counters,
bars and other instrumentation disagree while the physics agrees. This kit
packages the repairs the determinism ladder validated — the isolation-learned
chrome mask (43.2% → 76.3%) and an animation-mismatch exemption (→ 90.0%) —
as a dependency-free module small enough to ship into the python tool sandbox.

Contract (deliberately tiny):

    cm = ChromeMask(isolated=2)
    cm.learn(before, after)          # feed every observed transition
    cm.equal(a, b)                   # equality outside learned chrome
    cm.digest(board)                 # stable id of the masked board
    verdict = judge(predicted, observed, animation, cm)
        -> "match" | "mismatch" | "exempt"   (exempt: the animation shows the
           step passed through distinct intermediate states, so a settled-board
           disagreement is not evidence against the model)

No imports beyond hashlib; grids are sequences of sequences of ints, the shape
the sandbox already hands to model code.
"""
from __future__ import annotations

import hashlib


class ChromeMask:
    """Cells that change while nothing else does are instrumentation."""

    def __init__(self, isolated: int = 2) -> None:
        self._isolated = int(isolated)
        self.chrome: set[tuple[int, int]] = set()

    def learn(self, before, after) -> None:
        diff = [(r, c) for r, row in enumerate(before) for c, v in enumerate(row)
                if int(after[r][c]) != int(v)]
        if 0 < len(diff) <= self._isolated:
            self.chrome.update(diff)

    def equal(self, a, b) -> bool:
        for r, row in enumerate(a):
            brow = b[r]
            for c, v in enumerate(row):
                if (r, c) in self.chrome:
                    continue
                if int(v) != int(brow[c]):
                    return False
        return True

    def digest(self, board) -> str:
        rows = []
        for r, row in enumerate(board):
            rows.append(tuple(0 if (r, c) in self.chrome else int(v)
                              for c, v in enumerate(row)))
        return hashlib.blake2b(repr(tuple(rows)).encode(),
                               digest_size=8).hexdigest()


def animation_digest(frames, cm: ChromeMask) -> str:
    """One id for the whole film strip, chrome-masked frame by frame."""
    return hashlib.blake2b(
        "|".join(cm.digest(f) for f in frames).encode(),
        digest_size=8).hexdigest()


def judge(predicted, observed, animation, cm: ChromeMask,
          expected_animation: str | None = None) -> str:
    """Verdict for one prediction against one observed transition.

    `animation` is the observed film strip for the step (may be length 1);
    `expected_animation` is the animation digest recorded the last time this
    (state, action) was seen, if any. A settled-board mismatch is EXEMPT when
    the animations differ: §5 showed such steps pass through distinguishable
    intermediate states (all 49 of lf52's residual contradictions), so they
    are hidden-state transitions, not model errors.
    """
    if cm.equal(predicted, observed):
        return "match"
    if expected_animation is not None and animation is not None:
        if animation_digest(animation, cm) != expected_animation:
            return "exempt"
    return "mismatch"
