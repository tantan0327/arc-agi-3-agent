"""Stub test of the act-rule patch: the force line appears only after a turn without an action."""
from pathlib import Path
import inference.agent.tool_agent as TA  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
ns = {"__name__": "act_rule_patch"}
exec(compile((ROOT / "scripts/patches/act_rule.py").read_text(), "act_rule", "exec"), ns)
assert TA.ToolAgent._build_user_prompt is ns["_build_with_actrule"]
assert ns["SYSTEM_LINE"].strip() in TA._build_system_prompt(tool_output_tokens=3072)
print("1 hooks installed; system line present")
class Fake: pass
ns["_orig_build"] = lambda self, n, *a, **k: "opener %d" % n
ag = Fake()
t1 = ns["_build_with_actrule"](ag, 1, history_entries=["a"] * 3)            # first turn: no baseline
t2 = ns["_build_with_actrule"](ag, 2, history_entries=["a"] * 3)            # no action since -> streak 1 -> forced
t3 = ns["_build_with_actrule"](ag, 3, history_entries=["a"] * 5)            # acted -> reset
t4 = ns["_build_with_actrule"](ag, 4, history_entries=["a"] * 5)            # no action -> forced again
assert ns["FORCE_LINE"] not in t1 and ns["FORCE_LINE"] in t2 and ns["FORCE_LINE"] not in t3 and ns["FORCE_LINE"] in t4
print("2 force line only after a no-action turn; resets after an action")
print("OK")
