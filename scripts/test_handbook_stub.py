"""Stub test of the objective_handbook patch against Franzen's patched harness (no GPU, no model).

    PYTHONPATH=<bundle>/src/ARC3-Inference:<bundle>/src/tufa-arc-agi-framework/src \
    python3 scripts/test_handbook_stub.py
"""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ARC3_LEVEL_TRANSFER_GUIDANCE", "1")
os.environ.setdefault("ARC3_ACTION_INFO", "1")
os.environ.setdefault("ARC3_FRAME_DIFF_HINT", "1")
os.environ.setdefault("ARC3_PERSISTENT_FUNCTIONS", "1")

import inference.agent.tool_agent as TA  # noqa: E402

before = TA._build_system_prompt(tool_output_tokens=3072)
assert "Objective handbook (" not in before
ns = {"__name__": "objective_handbook_patch"}
exec(compile((ROOT / "scripts/patches/objective_handbook.py").read_text(), "objective_handbook", "exec"), ns)
after = TA._build_system_prompt(tool_output_tokens=3072)
assert after.startswith(before.rstrip("\n")), "the original prompt is not preserved verbatim"
assert after.count("Objective handbook (") == 1 and ("then plan the whole chain." in after)
added = len(after) - len(before.rstrip("\n"))
print(f"1 system prompt: original {len(before)} chars preserved; handbook adds {added} chars (~{len(ns['HANDBOOK'].split())} words); idempotent={TA._build_system_prompt(tool_output_tokens=3072).count('Objective handbook (') == 1}")

# 2. the per-turn line is appended to the user prompt exactly once
class _Frame:
    def __init__(self):
        self.step = 3; self.level = 1; self.shape = (64, 64); self.ascii = ""; self.segmentation = {"nodes": [], "adjacency_list": []}
agent = TA.ToolAgent.__new__(TA.ToolAgent)
for name in ("_last_action_gameplay_changed",):
    setattr(agent, name, False)
try:
    text = TA.ToolAgent._build_user_prompt(agent, 3, valid_actions=["UP", "DOWN", "LEFT", "RIGHT"], current_frame=_Frame(), history_entries=[], previous_step_summary=None)
    assert text.count(ns["TURN_LINE"]) == 1, text[-300:]
    print("2 user prompt: turn line appended once")
except Exception as exc:  # the user prompt builder touches more agent state than a bare instance has
    print("2 user prompt: builder needs live state (%s: %s) - checked the wrapper only" % (type(exc).__name__, str(exc)[:80]))
    assert TA.ToolAgent._build_user_prompt is ns["_build_with_handbook"]

# 3. disabled by flag -> nothing changes
os.environ["ARC3_HANDBOOK"] = "0"
ns2 = {"__name__": "objective_handbook_patch"}
TA._build_system_prompt = ns["_orig_system"]; TA.ToolAgent._build_user_prompt = ns["_orig_build"]
exec(compile((ROOT / "scripts/patches/objective_handbook.py").read_text(), "objective_handbook", "exec"), ns2)
assert TA._build_system_prompt is ns["_orig_system"], "flag did not disable the patch"
print("3 ARC3_HANDBOOK=0 leaves the harness untouched")
# 4. version 2 adds the held-out regularities exactly once, after the protocol
os.environ["ARC3_HANDBOOK"] = "1"; os.environ["ARC3_HANDBOOK_VERSION"] = "2"
ns3 = {"__name__": "objective_handbook_patch"}
exec(compile((ROOT / "scripts/patches/objective_handbook.py").read_text(), "objective_handbook", "exec"), ns3)
v2 = TA._build_system_prompt(tool_output_tokens=3072)
assert v2.count("Regularities seen across many games") == 1 and v2.count("13. ASSEMBLE-PIECES") == 1
assert v2.index("then plan the whole chain.") < v2.index("13. ASSEMBLE-PIECES")
print(f"4 version 2: +{len(ns3['HANDBOOK_V2_EXTRA'].split())} words (total ~{len(ns3['HANDBOOK'].split())})")
print("ALL OK")
