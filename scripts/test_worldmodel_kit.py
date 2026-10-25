"""v8-D1 acceptance: the kit reproduces the determinism ladder on recorded data.

No LLM involved. Loads a recorded run, finds genuine repeats of
(game, level, raw before-board, action), and scores after-board agreement
raw vs chrome-masked. The original §5 measurement (different run) gave
43.2% raw -> 76.3% masked; this asserts the *shape* (masked beats raw by a
wide margin), not the exact numbers, plus unit checks on judge().

    python scripts/test_worldmodel_kit.py <run_dir>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from duck_delta.worldmodel_kit import ChromeMask, judge  # noqa: E402


def unit_tests() -> None:
    cm = ChromeMask()
    a = [[1, 2], [3, 4]]
    b = [[1, 2], [3, 5]]
    assert not cm.equal(a, b)
    cm.learn(a, b)                       # single-cell change -> chrome
    assert cm.equal(a, b)
    assert cm.digest(a) == cm.digest(b)
    assert judge(a, b, None, cm) == "match"
    c = [[9, 9], [9, 9]]
    assert judge(a, c, None, cm) == "mismatch"
    from duck_delta.worldmodel_kit import animation_digest
    exp = animation_digest([a], cm)
    # b is chrome-equal to a, so its animation MATCHES the expectation ->
    # a settled mismatch stays a mismatch (no exemption without new evidence)
    assert judge(a, c, [b], cm, expected_animation=exp) == "mismatch"
    # c's animation differs beyond chrome -> the step demonstrably passed
    # through a different intermediate world: exempt
    assert judge(a, c, [c], cm, expected_animation=exp) == "exempt"
    print("unit tests: ok")


def ladder(run_dir: Path) -> None:
    from load_recorded_states import load
    import numpy as np

    games_states = load(str(run_dir / "intermediate_states.pkl"))
    raw_n = raw_agree = m_n = m_agree = 0
    for states in games_states:
        if len(states) < 3:
            continue
        cm = ChromeMask()
        seen: dict = {}
        boards = [np.array(s.raw.frame)[-1].tolist() for s in states]
        for i in range(1, len(states)):
            a = getattr(states[i], "previous_action", None)
            ar = getattr(getattr(a, "id", None), "_args", None)
            name = ar[1] if ar and len(ar) > 1 else None
            data = getattr(a, "data", None)
            coord = None
            if isinstance(getattr(data, "__dict__", None), dict):
                coord = (data.__dict__.get("y"), data.__dict__.get("x"))
            elif isinstance(data, dict):
                coord = (data.get("y"), data.get("x"))
            before, after = boards[i - 1], boards[i]
            cm.learn(before, after)
            key = (repr(before), name, coord)
            if key in seen:
                prev_after = seen[key]
                raw_n += 1
                raw_agree += prev_after == after
                m_n += 1
                m_agree += cm.equal(prev_after, after)
            else:
                seen[key] = after
    print(f"repeats: {raw_n}")
    print(f"raw agreement:    {raw_agree}/{raw_n} = {raw_agree/max(1,raw_n):.1%}")
    print(f"masked agreement: {m_agree}/{m_n} = {m_agree/max(1,m_n):.1%}")
    raw_err = raw_n - raw_agree
    m_err = m_n - m_agree
    assert m_err <= max(0, raw_err // 2), \
        f"mask should absorb at least half the raw errors ({raw_err} -> {m_err})"
    print(f"ladder shape: reproduced (errors {raw_err} -> {m_err})")


if __name__ == "__main__":
    unit_tests()
    ladder(Path(sys.argv[1]))
