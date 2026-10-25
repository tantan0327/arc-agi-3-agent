"""GP-C: can an open model AUTHOR a game's abstraction? Priced against GP-B's bar.

For each of the 22 replay games the model is shown a few frames and
transitions and asked to write `extract_state(board) -> tuple` -- the
game-specific state variables (Astra's "L8: hub q2. Lengths: 14=1"). The
authored code is scored offline against the human replays with GP-A's
DYNAMICS view, and the score plus concrete failure examples go back to the
model for revision (GP-B: every hand-written reference needed 2-3
scorer-driven fixes; one-shot authoring would be judged unfairly).

Runs anywhere with an OpenAI-compatible endpoint (the flash-next vLLM server
on Kaggle), or with --mock (returns the GP-B references) to test the loop.

    python scripts/gpc_author.py --mock --games m0r0,tr87,ar25 --rounds 2
    python scripts/gpc_author.py --base-url http://127.0.0.1:8000/v1 --rounds 3
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import re
import signal
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

# Kaggle auto-decompresses a .gz dataset file into replay_cache.pkl, so both
# spellings are accepted wherever the cache is looked for.
CACHE_CANDIDATES = [ROOT / "scratchpad" / "replay_cache.pkl.gz", ROOT / "scratchpad" / "replay_cache.pkl",
                    HERE / "replay_cache.pkl.gz", HERE / "replay_cache.pkl"]


def load_cache(path):
    if str(path).endswith(".gz"):
        return pickle.load(gzip.open(path))
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------- helpers
def blobs(board):
    """4-connected components per colour: {colour: [(size, top, left), ...]}.
    Offered to the authored code so perception is one call away."""
    g = np.asarray(board)
    out = {}
    seen = np.zeros(g.shape, dtype=bool)
    rows, cols = g.shape
    for r in range(rows):
        for c in range(cols):
            if not seen[r, c]:
                colour = int(g[r, c]); stack = [(r, c)]; seen[r, c] = True; cells = []
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols and not seen[ny, nx] and int(g[ny, nx]) == colour:
                            seen[ny, nx] = True; stack.append((ny, nx))
                out.setdefault(colour, []).append(
                    (len(cells), min(y for y, _ in cells), min(x for _, x in cells)))
    return out


def render(board):
    chars = "0123456789ABCDEF"
    return "\n".join("".join(chars[int(v) % 16] for v in row) for row in np.asarray(board))


def describe_diff(a, b):
    d = np.argwhere(np.asarray(a) != np.asarray(b))
    if len(d) == 0:
        return "no change"
    before = Counter(int(np.asarray(a)[y, x]) for y, x in d).most_common(3)
    after = Counter(int(np.asarray(b)[y, x]) for y, x in d).most_common(3)
    return (f"{len(d)} cells changed in rows {d[:,0].min()}-{d[:,0].max()} cols {d[:,1].min()}-{d[:,1].max()}; "
            f"colours before {before} -> after {after}")


def action_text(name, click):
    if click:
        return f"{name} (click in the 8x8 cell row-block {click[0]}, col-block {click[1]}, i.e. rows {click[0]*8}-{click[0]*8+7}, cols {click[1]*8}-{click[1]*8+7})"
    return name


# ------------------------------------------------------------------ prompt
CONTRACT = """You are writing PERCEPTION code for an unknown 64x64 grid game (cells are integers 0-15 = colours).
Write one Python function:

    def extract_state(board) -> tuple

`board` is a 64x64 list of lists of ints. Return a hashable tuple that captures the game's STATE VARIABLES the way a careful player would note them: which level/layout this is (e.g. a hash of the static walls), and the position / size / count of everything that moves or changes in response to actions (an avatar, a cursor, bars that extend, tiles that flip, items collected). Use plain ints and nested tuples.

Rules that the scorer will enforce:
- DETERMINISTIC: the same board must always give the same tuple (no randomness, no global state).
- RESPONSIVE: when an action visibly changes the board, the tuple should change too -- do not throw away the moving parts.
- CLEAN: exclude decorations and step/energy counters that tick on every action regardless of what happened (they usually live on the border or in a corner); including them makes every state unique and useless.
- NUMERIC where things extend or move: lengths and positions as ints, so the same action produces the same delta.
- Available in scope: `np` (numpy) and `blobs(board) -> {colour: [(size, top, left), ...]}` (4-connected components per colour). No imports needed; do not read files or use randomness.

Keep it SHORT: under 80 lines, no docstrings or commentary inside the code, no example usage. Do not reassign the name `blobs`.

Return ONLY the Python code for extract_state (and any small helpers), inside one ```python block."""


def mechanical_hints(decisions):
    """What the harness can tell the model for free, from the recorded play.
    (1) INSTRUMENTATION cells, by the determinism ladder's ISOLATION rule: a
        cell is chrome if it changes in a transition where (almost) nothing
        else does (<=2 cells) -- counters, bars, blinking indicators. Per-cell
        change *frequency* misses a filling bar (each cell flips once), which
        is why v6's hint was empty on m0r0.
    (2) Which colours' cells ever changed within a level vs never did, computed
        over WITHIN-LEVEL transitions only (a level change redraws the whole
        board and would make every colour look dynamic)."""
    boards = [np.asarray(b) for b, _, _ in decisions]
    n = len(boards) - 1
    if n < 5:
        return ""
    chrome = np.zeros(boards[0].shape, dtype=bool)
    count = np.zeros(boards[0].shape)
    within = 0
    moved = set(); present = set()
    for i in range(n):
        d = boards[i] != boards[i + 1]
        k = int(d.sum())
        if k == 0:
            continue
        if k <= 2:
            chrome |= d
        if k > 0.3 * d.size:          # a redraw / level change: not a within-level move
            continue
        within += 1
        count += d
        present.update(np.unique(boards[i]).tolist())
        moved.update(np.unique(boards[i][d]).tolist()); moved.update(np.unique(boards[i + 1][d]).tolist())
    if within:
        chrome |= (count / within) > 0.3   # a cell that flips on >30% of moves is an indicator too
    lines = []
    cells = np.argwhere(chrome)
    if len(cells):
        rows = sorted(set(cells[:, 0].tolist())); cols = sorted(set(cells[:, 1].tolist()))
        lines.append(f"- INSTRUMENTATION: {len(cells)} cells in rows {rows[0]}-{rows[-1]}, cols {cols[0]}-{cols[-1]} change on their own (a step counter / energy bar / indicator). They are NOT game state -- exclude them completely, e.g. ignore those rows/cols.")
    else:
        lines.append("- No isolated counter/bar cells were detected; every change is a game move.")
    static = sorted(int(c) for c in present if c not in moved)
    mv = sorted(int(c) for c in moved)
    lines.append(f"- Within a level, colours whose cells changed: {mv}. Colours that never changed within a level (static layout -- use only as a level identity, e.g. a hash of where they are): {static}.")
    return "\nMECHANICAL FACTS measured from the recorded play (trust these):\n" + "\n".join(lines)


def build_prompt(gid, decisions, feedback=None, n_full=3, n_diffs=8):
    parts = [CONTRACT, mechanical_hints(decisions), f"\nGAME {gid[:4]}. The initial board:\n```\n{render(decisions[0][0])}\n```"]
    # a few full transitions spread over the first ~40 decisions, then diff-only ones
    span = min(len(decisions) - 1, 40)
    idx_full = sorted(set([0, span // 3, (2 * span) // 3][:n_full]))
    shown = set()
    for i in idx_full:
        b, name, click = decisions[i]; nb = decisions[i + 1][0]
        parts.append(f"\nAfter action {action_text(name, click)} at step {i}: {describe_diff(b, nb)}. Board after:\n```\n{render(nb)}\n```")
        shown.add(i)
    diffs = []
    step = max(1, span // n_diffs)
    for i in range(0, span, step):
        if i in shown or len(diffs) >= n_diffs:
            continue
        b, name, click = decisions[i]; nb = decisions[i + 1][0]
        diffs.append(f"- step {i}: {action_text(name, click)} -> {describe_diff(b, nb)}")
    if diffs:
        parts.append("\nMore transitions (diff summaries only):\n" + "\n".join(diffs))
    if feedback:
        parts.append("\nYOUR PREVIOUS extract_state WAS SCORED AGAINST RECORDED HUMAN PLAY. Fix the problems below and return the full revised function.\n" + feedback)
    return "\n".join(parts)


# ----------------------------------------------------------------- scoring
class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def compile_fn(code):
    start = code.find("```python")
    if start >= 0:
        body = code[start + len("```python"):]
        end = body.rfind("```")
        src = body[:end] if end >= 0 else body
    else:
        src = code
    ns = {"np": np, "blobs": blobs}
    exec(compile(src, "<authored>", "exec"), ns)
    fn = ns.get("extract_state")
    if fn is None:
        raise ValueError("no extract_state defined")
    return fn, src


def score_fn(fn, gid, decisions, budget_s=120.0):
    """GP-A dynamics view + concrete failure examples for feedback."""
    keys = []
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    for board, _, _ in decisions:
        if time.time() - t0 > budget_s:
            raise Timeout(f"scoring exceeded {budget_s}s")
        signal.alarm(5)
        try:
            k = fn(board.tolist())
            hash(k)
        finally:
            signal.alarm(0)
        keys.append(k)
    trans = defaultdict(Counter); seen_at = defaultdict(list)
    sa = det = resp = 0
    blind = []; nondet = defaultdict(list)
    for i in range(len(decisions) - 1):
        board, name, click = decisions[i]
        akey = (name, click) if click else name
        key, nxt = keys[i], keys[i + 1]
        changed = not np.array_equal(decisions[i][0], decisions[i + 1][0])
        if nxt != key:
            resp += 1
        elif changed and len(blind) < 3:
            blind.append(f"  step {i}: {action_text(name, click)} changed the board ({describe_diff(decisions[i][0], decisions[i+1][0])}) but your state did not change")
        tc = trans[(key, akey)]
        if tc:
            sa += 1
            if tc.most_common(1)[0][0] == nxt:
                det += 1
            else:
                nondet[(key, akey)].append(i)
        tc[nxt] += 1
        seen_at[(key, akey)].append(i)
    n = len(decisions) - 1
    metrics = {"sa_cov": sa / n, "determ": det / sa if sa else 0.0, "respons": resp / n,
               "states": len(set(keys)), "decisions": len(decisions)}
    # Which parts of the tuple change on nearly every step? Those are the
    # counters/decorations that make every state unique (v3's dominant failure).
    def leaves(k, path=""):
        if isinstance(k, (tuple, list)):
            out = []
            for i, v in enumerate(k):
                out.extend(leaves(v, f"{path}[{i}]"))
            return out
        return [(path, k)]
    flat = [leaves(k) for k in keys]
    volatile = []
    if flat and all(len(f) == len(flat[0]) for f in flat) and len(flat[0]) > 1:
        L = len(flat[0]); changes = [0] * L
        for i in range(1, len(flat)):
            for j in range(L):
                if flat[i][j][1] != flat[i - 1][j][1]:
                    changes[j] += 1
        ranked = sorted(range(L), key=lambda j: -changes[j])
        for j in ranked[:3]:
            rate = changes[j] / (len(flat) - 1)
            if rate > 0.6:
                sample = [flat[i][j][1] for i in range(0, min(len(flat), 5))]
                volatile.append(f"  tuple element {flat[0][j][0]} changes on {rate:.0%} of steps (values e.g. {sample}) -- a counter or decoration: remove it, or coarsen it if it is a genuine moving part")
    metrics["volatile"] = volatile
    examples = []
    for (key, akey), steps in list(nondet.items())[:2]:
        first = seen_at[(key, akey)][0]; j = steps[0]
        examples.append(f"  same state + action {akey if isinstance(akey, str) else akey[0]} at steps {first} and {j} gave different next states; the two boards differ: {describe_diff(decisions[first][0], decisions[j][0])}")
    return metrics, blind, examples


def grade(m):
    """GP-B bar: planner-grade if determ>=0.9 at sa_cov>0.10 (revisiting games);
    a progression game shows respons high with sa_cov ~0, judged on delta rules (not here)."""
    if m["respons"] < 0.5:
        return "BLIND"
    if m["sa_cov"] > 0.10 and m["determ"] >= 0.90:
        return "PASS"
    if m["sa_cov"] < 0.03:
        return "NO-REVISIT"
    return "NONDET" if m["determ"] < 0.9 else "LOW-COV"


def feedback_text(m, blind, examples, error=None):
    if error:
        return (f"Your previous reply could not be used: {error}. Return ONLY the complete "
                "extract_state function in one ```python block, with no explanation before it.")
    lines = [f"Scores: next-state determinism {m['determ']:.0%} over {m['sa_cov']:.0%} of decisions at a previously seen (state, action); "
             f"state changed after {m['respons']:.0%} of actions; {m['states']} distinct states over {m['decisions']} boards. Verdict: {grade(m)}."]
    if grade(m) == "BLIND":
        lines.append("Your state ignores most moves. Include the things that move:")
    lines += blind
    if examples:
        lines.append("Non-determinism (your state conflates two situations that behave differently -- add the variable that distinguishes them):")
        lines += examples
    if grade(m) == "NO-REVISIT":
        lines.append("Your state almost never repeats, so nothing can be learned from it. The parts below change on nearly every step; remove them (they are counters/decorations) or coarsen them:")
        lines += m.get("volatile") or ["  (could not attribute -- return a flat tuple of named parts so the scorer can tell you which one)"]
        lines.append("Keep ONLY: a level/layout identity and the position/size of the few things the player controls or must reach.")
    return "\n".join(lines)


# ------------------------------------------------------------------- model
def call_model(base_url, model, prompt, max_tokens=12000, temperature=0.2, timeout=900):
    """Qwen3-class servers run a reasoning parser: the model's thinking lands in
    `reasoning_content` and, if it exhausts max_tokens while thinking, `content`
    is None (GP-C v2 died on exactly that). Thinking is switched off for this
    task via chat_template_kwargs, the budget raised, and whatever text came
    back is returned -- never None."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": temperature,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    ch = out["choices"][0]
    msg = ch.get("message") or {}
    text = msg.get("content") or msg.get("reasoning_content") or ""
    fin = ch.get("finish_reason")
    if not text:
        return f"[EMPTY RESPONSE finish_reason={fin}]"
    if fin == "length" and "```" not in text:
        return text + "\n[TRUNCATED finish_reason=length]"
    return text


def discover(base_url):
    if base_url:
        cands = [base_url]
    else:
        cands = [os.environ.get(k) for k in ("LOCAL_ANALYZER_BASE_URL", "OPENAI_BASE_URL") if os.environ.get(k)]
        cands += [f"http://127.0.0.1:{p}/v1" for p in (1234, 8000, 8001, 30000, 8080)]
    for u in cands:
        try:
            with urllib.request.urlopen(u.rstrip("/") + "/models", timeout=10) as r:
                data = json.loads(r.read())
            ids = [m["id"] for m in data.get("data", [])]
            if ids:
                return u, ids[0]
        except Exception:
            continue
    raise SystemExit("no OpenAI-compatible server found")


MOCK_SRC = {}


def mock_model(gid, prompt):
    src = (HERE / "gpb_reference_abstractions.py").read_text()
    name = gid[:4]
    if name in ("m0r0", "tr87", "s5i5"):
        body = src.split("def " + name + "(board):", 1)[1]
        # cut at the next top-level def / assignment
        body = re.split(r"\n(?=def |ABSTRACTIONS)", body, 1)[0]
        helpers = ""
        if name == "m0r0":
            helpers = "def _blobs(mask):\n    import numpy as np\n    out=[]\n    for c,v in blobs(np.where(mask,1,0).tolist()).items():\n        if c==1: out += [(s,t,l) for s,t,l in v]\n    return out\n"
        if name == "s5i5":
            helpers = "def _components_by_colour(grid):\n    return blobs(np.asarray(grid).tolist())\n"
        return f"```python\nimport numpy as np\n{helpers}def extract_state(board):{body}\n```"
    return "```python\ndef extract_state(board):\n    b = blobs(board)\n    return tuple(sorted((c, len(v)) for c, v in b.items()))\n```"


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url"); ap.add_argument("--model")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--games", help="comma-separated 4-char prefixes")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=1, help="best-of-N candidates on round 0 (scorer selects)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--out", default=str(ROOT / "scratchpad" / "gpc_results.json"))
    args = ap.parse_args()
    cache = next((c for c in CACHE_CANDIDATES if c.exists()), None)
    if cache is None:
        sys.exit("replay cache missing")
    games = load_cache(cache)
    if args.games:
        want = set(args.games.split(","))
        games = [(g, d) for g, d in games if g[:4] in want]
    if not args.mock:
        base_url, model = discover(args.base_url)
        model = args.model or model
        print(f"server {base_url} model {model}", flush=True)
        smoke = call_model(base_url, model, "Reply with exactly: ```python\nprint(1)\n```", max_tokens=200)
        print(f"smoke reply ({len(smoke)} chars): {smoke[:200]!r}", flush=True)
    results = {}
    for gid, decisions in games:
        feedback = None; best = None; history = []
        GRADE_RANK = {"PASS": 5, "LOW-COV": 4, "NONDET": 3, "NO-REVISIT": 2, "BLIND": 1, "ERROR": 0}

        def rank(row):
            m = row["metrics"] or {"determ": 0, "sa_cov": 0, "respons": 0}
            return (GRADE_RANK.get(row["grade"], 0), m["determ"] * min(1.0, m["sa_cov"] / 0.1), m["respons"])

        def attempt(rnd, prompt, temperature):
            t0 = time.time()
            try:
                reply = mock_model(gid, prompt) if args.mock else call_model(base_url, model, prompt, temperature=temperature)
            except Exception as e:  # noqa: BLE001 - a failed request must not kill the run
                reply = f"[REQUEST FAILED {type(e).__name__}: {str(e)[:200]}]"
            t_model = time.time() - t0
            reply = reply if isinstance(reply, str) else str(reply)
            try:
                fn, src = compile_fn(reply)
                metrics, blind, examples = score_fn(fn, gid, decisions)
                error = None
            except Exception as e:  # noqa: BLE001
                metrics, blind, examples, src = None, [], [], reply[:2000]
                error = f"{type(e).__name__}: {str(e)[:300]}"
                if reply.startswith("["):
                    error = reply[:300]
            row = {"round": rnd, "metrics": metrics, "grade": grade(metrics) if metrics else "ERROR",
                   "error": error, "code": src, "model_s": round(t_model, 1), "prompt_chars": len(prompt),
                   "blind": blind, "examples": examples}
            print(f"{gid[:4]} r{rnd}: {row['grade']:10s} " + (f"determ {metrics['determ']:.0%} sa_cov {metrics['sa_cov']:.0%} respons {metrics['respons']:.0%} states {metrics['states']}" if metrics else (error or '')[:80]) + f"  [{t_model:.0f}s]", flush=True)
            return row

        for rnd in range(args.rounds):
            prompt = build_prompt(gid, decisions, feedback)
            n = args.samples if rnd == 0 else 1
            cands = [attempt(rnd, prompt, args.temperature if n > 1 else 0.2) for _ in range(n)]
            row = max(cands, key=rank)
            history.extend(cands)
            if best is None or rank(row) > rank(best):
                best = row
            if best["grade"] == "PASS":
                break
            feedback = feedback_text(best["metrics"], best["blind"], best["examples"], best["error"]) if (best["metrics"] or best["error"]) else None
        results[gid] = {"best": best, "history": history}
        Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    print("\nSUMMARY (best round per game):")
    for gid, r in results.items():
        b = r["best"]
        print(f"  {gid[:4]} {b['grade']:10s} determ {b['metrics']['determ']:.0%} sa_cov {b['metrics']['sa_cov']:.0%} respons {b['metrics']['respons']:.0%} states {b['metrics']['states']} (round {b['round']})" if b else f"  {gid[:4]} all rounds failed")
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
