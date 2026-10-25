"""Stub test of the priority_form patch against Franzen's patched harness (no GPU, no model).

    PYTHONPATH=<bundle>/src/ARC3-Inference:<bundle>/src/tufa-arc-agi-framework/src \
    ARC3_PRIORITY_FORM=dprime python3 scripts/test_priority_form_stub.py
"""
from __future__ import annotations
import importlib.util, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ARC3_PRIORITY_FORM", "dprime")
os.environ.setdefault("ARC3_MAX_ACTIVE_STREAMS", "10")
os.environ["ARC3_PRIORITY_SCORE_NORMALIZATION"] = "1"
os.environ["ARC3_PRIORITY_TAIL_LOOKUP"] = "remaining"

import inference.agent.tool_agent as TA  # noqa: E402
import inference.framework.solver as SV  # noqa: E402

orig_pv = TA.priority_value
ns = {"__name__": "priority_form_patch"}
exec(compile((ROOT / "scripts/patches/priority_form.py").read_text(), "priority_form", "exec"), ns)
assert TA.priority_value is ns["priority_value"], "priority_value not replaced"
assert TA.ProgressPace is ns["PaceTracker"], "ProgressPace not replaced"
assert getattr(TA._PriorityGate.acquire, "_priority_form", False), "acquire not wrapped"
assert getattr(SV._HarnessGameSession.play, "_priority_form", False), "play not wrapped"
assert os.environ["ARC3_PRIORITY_PACE"] == "1" and os.environ["ARC3_PRIORITY_TAIL_FADE"] == "1"
print("1 installed: priority_value, ProgressPace, acquire, play; env set")

# 2. the gate prices snapshots through OUR function, with tail fade from its clock
gate = TA._PriorityGate(10)
now = time.monotonic()
gate.configure_clock(now, now + 9 * 3600)
S = TA.PrioritySnapshot
fresh = gate._snapshot_priority(S(1, 0, 0.0, 0.0, 8), now)
deep = gate._snapshot_priority(S(4, 10, 20000.0, 15000.0, 8), now)
stuck = gate._snapshot_priority(S(2, 200, 400000.0, 0.0, 8), now)
assert deep > fresh > stuck, (deep, fresh, stuck)
late = gate._snapshot_priority(S(1, 0, 0.0, 0.0, 8), now + 9 * 3600 - 60)   # tail faded -> B ~ 0
assert late < fresh, (late, fresh)
print("2 ordering: deep progressing > fresh > stuck; tail fades the room-to-grow bonus")

# 3. the pace tracker reports mean tokens per cleared level (what the forms read)
p = TA.ProgressPace(); assert p.cost_multiplier() == 0.0
p.record_completion(20000, 45000); p.record_completion(40000, 45000)
assert abs(p.cost_multiplier() - 30000) < 1e-6
print("3 pace tracker = mean tokens per cleared level")

# 4. fresh pricing: a fresh acquire goes through the formula, not the 2,000,000 band
TA.__dict__  # noqa
class FakeGate(TA._PriorityGate):
    def __init__(self): super().__init__(10); self.seen = []
    def _enqueue_and_wait(self, priority, snapshot=None):
        self.seen.append((priority, snapshot)); self._admitted.add(self._seq + 1)
g = FakeGate(); g.configure_clock(now, now + 9 * 3600)
ns["_TL"].total_levels = 7
g.acquire(TA._PRIORITY_UNTRIMMED_BASE - 5)
assert g.seen and g.seen[0][0] < TA._PRIORITY_UNTRIMMED_BASE - TA._PRIORITY_BAND and g.seen[0][1].total_levels == 7, g.seen
print("4 fresh games are priced by the formula (with the game's level count)")

# 5. formulas agree with the simulator's copies on a grid
spec = importlib.util.spec_from_file_location("ss", ROOT / "scripts/sched_sim.py"); ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
worst = 0.0
for level in (1, 2, 4, 6):
    for a in (0, 10, 60):
        for t in (0.0, 30000.0, 200000.0):
            for pace in (0.0, 12000.0, 50000.0):
                for N in (6, 8, 10):
                    for phi in (1.0, 0.3):
                        d1 = ns["value_dprime"](level, a, t, pace, N, phi); d2 = ss.dprime_value(level, a, t, pace, N, phi) / 1000.0
                        o1 = ns["value_ours"](level, a, t, pace, N, phi, ns["OURS_DEFAULT"]); o2 = ss.ours_value(level, a, t, pace, N, phi, ss.OURS) / 1000.0
                        worst = max(worst, abs(d1 - d2), abs(o1 - o2))
assert worst < 1e-9, worst
print("5 patch formulas == simulator formulas (max diff %.1e)" % worst)

# 6. the installer refuses a changed harness
class Dummy: pass
try:
    ns["install"](Dummy(), SV); raise AssertionError("should have raised")
except RuntimeError as exc:
    assert "harness version changed" in str(exc)
print("6 installer raises on a changed harness")
print("OK")
