"""Adoption and deliberation-cost read for prompt/tool arms on the Franzen chassis.

    python3 scripts/read_handbook_adoption.py scratchpad/arc3-franzen-handbook_out [more dirs]

Counts vocabulary ONLY in the model's own text ([THINKING] and [ASSISTANT] blocks of the transcripts;
the [SYSTEM PROMPT] block is repeated on every turn and would inflate any naive count), and reads the
deliberation cost per game from the requests files: turns, completion tokens, actions per turn,
completion tokens per action and per cleared level. Band dirs are printed for comparison.
"""
from __future__ import annotations
import collections, glob, json, os, re, sys

ARCHETYPES = ["NAVIGATE-TO-TARGET", "COLLECT-THEN-EXIT", "KEY-AND-DOOR", "PUSH-BLOCKS", "MATCH-REFERENCE-PATTERN",
              "TOGGLE-SWITCHES", "SEQUENCE-ORDER", "ROUTE-BEAM", "SORT-OR-ALIGN", "COUNTER-OR-RESOURCE", "GROW-OR-FILL",
              "SELECT-BY-PROPERTY", "ASSEMBLE-PIECES", "REDUCE-TO-ONE", "MEET-AND-MERGE"]
TOOLS = ["motion_theory(", "predict_move(", "path_to("]
BAND_DIRS = ["scratchpad/arc3-copy-franzen-m2_out", "scratchpad/arc3-franzen-rep_out", "scratchpad/arc3-franzen-rep2_out"]
BLOCK_RE = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|THINKING|ASSISTANT|MODEL RESPONSE META|ANALYZER STATUS)\]\s*$", re.M)


def model_text(txt: str) -> str:
    """Concatenate the [THINKING] and [ASSISTANT] blocks only."""
    out = []
    parts = BLOCK_RE.split(txt)
    # parts = [pre, name1, body1, name2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        if parts[i] in ("THINKING", "ASSISTANT"):
            out.append(parts[i + 1])
    return "\n".join(out)


def completion_tokens(path: str):
    tot = 0; n = 0
    for line in open(path, errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        n += 1
        u = None
        for holder in (r, r.get("response"), r.get("event")):
            if isinstance(holder, dict) and isinstance(holder.get("usage"), dict):
                u = holder["usage"]; break
        if isinstance(u, dict) and "completion_tokens" in u:
            tot += int(u["completion_tokens"] or 0)
        else:
            # fall back: search the record for a completion_tokens field
            s = line
            m = re.search(r'"completion_tokens":\s*(\d+)', s)
            if m:
                tot += int(m.group(1))
    return n, tot


def read(d):
    per = {}
    p = os.path.join(d, "benchmark.json")
    if os.path.exists(p):
        for r in json.load(open(p))["game_runs"]:
            g = r["game_id"][:4]
            per[g] = {"levels": int(r.get("levels_completed") or 0), "actions": len(r.get("history") or []), "state": r.get("state")}
    for f in glob.glob(os.path.join(d, "transcripts", "*_p0.txt")):
        g = os.path.basename(f)[:4]
        body = model_text(open(f, errors="replace").read())
        rec = per.setdefault(g, {"levels": None, "actions": None})
        counts = collections.Counter({a: len(re.findall(re.escape(a), body)) for a in ARCHETYPES})
        rec["archetype_mentions"] = sum(counts.values())
        rec["top_archetype"] = counts.most_common(1)[0][0] if sum(counts.values()) else "-"
        rec["hypothes"] = len(re.findall(r"hypothes", body, re.I))
        rec["tool_calls"] = sum(len(re.findall(re.escape(t), body)) for t in TOOLS)
        rec["model_chars"] = len(body)
    for f in glob.glob(os.path.join(d, "*_requests.jsonl")):
        g = os.path.basename(f)[:4]
        n, tot = completion_tokens(f)
        per.setdefault(g, {})["turns"] = n
        per[g]["completion_tokens"] = tot
    return per


def main():
    dirs = sys.argv[1:] or ["scratchpad/arc3-franzen-handbook_out", "scratchpad/arc3-franzen-handbook2_out"]
    for d in dirs + BAND_DIRS:
        per = read(d)
        if not per:
            print(f"{d}: nothing to read"); continue
        lv = sum(v.get("levels") or 0 for v in per.values()); ac = sum(v.get("actions") or 0 for v in per.values())
        tn = sum(v.get("turns") or 0 for v in per.values()); ct = sum(v.get("completion_tokens") or 0 for v in per.values())
        am = sum(v.get("archetype_mentions") or 0 for v in per.values()); hy = sum(v.get("hypothes") or 0 for v in per.values())
        tc = sum(v.get("tool_calls") or 0 for v in per.values())
        print(f"\n{os.path.basename(d)}: levels {lv} actions {ac} turns {tn} completion tokens {ct/1000:.0f}k | actions/turn {ac/tn if tn else 0:.2f} tokens/action {ct/ac if ac else 0:.0f} tokens/level {ct/lv/1000 if lv else 0:.1f}k | model text: archetypes {am} hypothes {hy} motion-tool calls {tc}")
        for g in sorted(per):
            v = per[g]; ct_g = v.get("completion_tokens") or 0; tn_g = v.get("turns") or 0; ac_g = v.get("actions") or 0
            print(f"  {g}: L{v.get('levels')} a{ac_g} t{tn_g} tok{ct_g/1000:.0f}k a/t {ac_g/tn_g if tn_g else 0:.2f} {v.get('state','')[:8]:<8} | archetypes {v.get('archetype_mentions', 0)} ({v.get('top_archetype', '-')}) hypothes {v.get('hypothes', 0)} tools {v.get('tool_calls', 0)}")


if __name__ == "__main__":
    main()
