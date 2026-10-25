"""GP-A: score an abstraction against the human replays.

The Grand-Prize architecture (docs/grand-prize-strategy.md §3) stands or
falls on one question: how good is a given `extract_state(board) -> key`?
This scorer answers it with the only 100%-optimal data in existence — the
6,512 human decisions — using ONLINE NEXT-ACTION PREDICTION:

Walk each game's human trajectory in order, keeping a lookup table
`abstract state -> action counts`. At every decision, if the current
abstract state has been seen before, predict the modal past action and check
it against what the human actually did. Three numbers fall out:

- coverage   — fraction of decisions at an already-seen abstract state
               (too-fine abstractions fail here: A0 board = 1.7%)
- precision  — accuracy of the modal prediction where one exists
               (too-coarse abstractions fail here: conflated states demand
               different actions)
- predicted  — coverage x precision = the share of ALL human decisions a
               pure state-lookup replays correctly. THE score. A good
               abstraction is exactly one that makes this large.

`states` (distinct abstract states) is reported as a compactness indicator.
Two action granularities: `base` (action name only) and `click` (name plus
the clicked 8x8 cell — humans click board-conditioned targets, so a strong
abstraction should make click targets predictable too).

Usage:
    python scripts/score_abstraction.py --rebuild-cache   # once, ~1 min
    python scripts/score_abstraction.py                   # built-in baselines
    python scripts/score_abstraction.py --module my_abs.py
        # my_abs.py defines extract_state(board) -> hashable
        #   and/or ABSTRACTIONS = {game_id_prefix: callable}
        # NOTE: the module is exec'd — review model-authored code first.

The replay cache (boards + actions) lives in scratchpad/replay_cache.pkl.gz;
scoring runs are then pure CPU over arrays, seconds per abstraction.

GP-B calibration (2026-09-09) added the DYNAMICS view (see score()) and
taught three things: (1) the policy view is bounded by plan-phase -- humans
take different actions at identical states (m0r0: precision 42% at a maze
position, because they pass the same cell outbound and inbound) -- so
imitation-predictability is NOT the planner's criterion; (2) use
--granularity click for click games, otherwise every click is one action and
dynamics look nondeterministic; (3) some games never revisit a state (s5i5:
bars only grow), so lookup dynamics are useless there and the abstraction
must expose numeric variables whose DELTAS are constant per control (see
scripts/gpb_reference_abstractions.py, s5i5 docstring).
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "scratchpad" / "replay_cache.pkl.gz"
REPLAYS = ROOT / "vendor/taaf-kaggle-source/src/re-arc-3/re_arc/dsl/official_human_replays"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/taaf-kaggle-source/src/re-arc-3"))

from duck_delta.worldmodel_kit import ChromeMask  # noqa: E402


def rebuild_cache() -> None:
    from re_arc import EnvSampler
    from arcengine import GameAction
    import numpy as np
    by_value = {m.value: m for m in GameAction}
    games = []
    for p in sorted(REPLAYS.glob("*.json")):
        data = json.loads(p.read_text())
        gid = data["game_id"]
        try:
            env = EnvSampler(include=[gid]).make(gid)
        except Exception:
            continue
        obs = env.reset()
        decisions = []
        for actions in data["levels"]:
            for value, adata in actions:
                ga = by_value.get(int(value))
                if ga is None:
                    break
                board = np.array(obs.frame[-1], dtype=np.int8)
                click = None
                if adata and isinstance(adata, dict) and "x" in adata:
                    click = (int(adata["y"]) // 8, int(adata["x"]) // 8)
                decisions.append((board, ga.name, click))
                obs, _r, _d, _i = env.step(ga, adata or None)
        games.append((gid, decisions))
    CACHE.parent.mkdir(exist_ok=True)
    with gzip.open(CACHE, "wb") as f:
        pickle.dump(games, f)
    n = sum(len(d) for _, d in games)
    print(f"cached {len(games)} games, {n} decisions -> {CACHE}")


def _components(board):
    """(colour -> [(size, cy, cx)]) with 4-connectivity, numpy-free walk."""
    import numpy as np
    grid = np.asarray(board)
    out: dict[int, list] = defaultdict(list)
    seen = np.zeros(grid.shape, dtype=bool)
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if not seen[r, c]:
                colour = int(grid[r, c])
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols \
                                and not seen[ny, nx] and int(grid[ny, nx]) == colour:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                ys = sum(y for y, _ in cells) // len(cells)
                xs = sum(x for _, x in cells) // len(cells)
                out[colour].append((len(cells), ys, xs))
    return out


def a1_object_set(board):
    comps = _components(board)
    return tuple(sorted((c, len(v)) for c, v in comps.items()))


def a2_object_layout(board):
    comps = _components(board)
    return tuple(sorted((c, cy // 4, cx // 4)
                        for c, v in comps.items() for _, cy, cx in v))


def a3_census_avatar(board):
    comps = _components(board)
    rare = None
    for c, v in comps.items():
        for size, cy, cx in v:
            if size <= 6:
                cand = (len(v), size, c, cy, cx)
                if rare is None or cand < rare:
                    rare = cand
    census = tuple(sorted((c, sum(s for s, _, _ in v))
                          for c, v in comps.items()))
    return (census, (rare[3], rare[4]) if rare else None)


BASELINES = {
    "A0 masked board": "cm",     # special-cased: stateful chrome mask digest
    "A1 object set": a1_object_set,
    "A2 object layout": a2_object_layout,
    "A3 census+avatar": a3_census_avatar,
}


def score(fn, games, granularity="base"):
    """Two views of an abstraction, both from online walks of the replays.

    POLICY view (imitation): predict the human's next action from state.
      coverage / precision / predicted  (see module docstring)
    DYNAMICS view (what a planner needs): predict the NEXT abstract state
    from (state, action).
      sa_cov   fraction of decisions whose (state, action) was seen before
      determ   among those, fraction whose outcome matches the modal outcome
      respons  fraction of decisions after which the abstract state changed
               (a too-coarse abstraction is blind to most moves)
    A planner-grade abstraction has high determ at usable sa_cov, respons
    near the board-change rate, and few states.
    """
    tot = cov = hit = 0
    sa_cov = det_hit = resp = 0
    n_states = 0
    per_game = []
    for gid, decisions in games:
        table: dict = defaultdict(Counter)
        trans: dict = defaultdict(Counter)      # (key, akey) -> next key counts
        cm = ChromeMask() if fn == "cm" else None
        prev = None
        keys = []
        for board, name, click in decisions:
            b = board.tolist()
            if cm is not None:
                if prev is not None:
                    cm.learn(prev, b)
                keys.append(cm.digest(b))
                prev = b
            else:
                keys.append(fn(b))
        g_tot = g_cov = g_hit = g_sa = g_det = g_resp = 0
        for i, (board, name, click) in enumerate(decisions):
            key = keys[i]
            akey = (name, click) if granularity == "click" and click else name
            tot += 1; g_tot += 1
            counts = table[key]
            if counts:
                cov += 1; g_cov += 1
                if counts.most_common(1)[0][0] == akey:
                    hit += 1; g_hit += 1
            counts[akey] += 1
            if i + 1 < len(keys):
                nxt = keys[i + 1]
                if nxt != key:
                    resp += 1; g_resp += 1
                tc = trans[(key, akey)]
                if tc:
                    sa_cov += 1; g_sa += 1
                    if tc.most_common(1)[0][0] == nxt:
                        det_hit += 1; g_det += 1
                tc[nxt] += 1
        n_states += len(table)
        per_game.append({"gid": gid[:4], "coverage": g_cov / g_tot,
                         "precision": g_hit / g_cov if g_cov else 0,
                         "predicted": g_hit / g_tot,
                         "sa_cov": g_sa / g_tot,
                         "determ": g_det / g_sa if g_sa else 0,
                         "respons": g_resp / g_tot, "states": len(table)})
    return {"coverage": cov / tot, "precision": hit / cov if cov else 0.0,
            "predicted": hit / tot, "states": n_states,
            "sa_cov": sa_cov / tot, "determ": det_hit / sa_cov if sa_cov else 0.0,
            "respons": resp / tot, "per_game": per_game, "decisions": tot}


def load_module(path):
    import numpy  # noqa: F401 - available to authored code
    ns: dict = {}
    exec(compile(Path(path).read_text(), path, "exec"), ns)
    if "ABSTRACTIONS" in ns:
        per_game = ns["ABSTRACTIONS"]
        def dispatch_factory(gid):
            for prefix, f in per_game.items():
                if gid.startswith(prefix):
                    return f
            return ns.get("extract_state")
        return dispatch_factory
    return lambda gid: ns["extract_state"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--module", help="python file with extract_state/ABSTRACTIONS")
    ap.add_argument("--granularity", choices=["base", "click"], default="base")
    args = ap.parse_args()
    if args.rebuild_cache:
        rebuild_cache()
        return
    if not CACHE.exists():
        sys.exit("no cache - run with --rebuild-cache first")
    with gzip.open(CACHE, "rb") as f:
        games = pickle.load(f)

    def report(label, res):
        print(f"{label:<18} policy: cov {res['coverage']:5.1%} prec {res['precision']:5.1%} "
              f"PRED {res['predicted']:5.1%} | dynamics: sa_cov {res['sa_cov']:5.1%} "
              f"DETERM {res['determ']:5.1%} respons {res['respons']:5.1%} | states {res['states']}")

    if args.module:
        factory = load_module(args.module)
        # score with per-game dispatch: wrap as a closure over gid
        tot_res = None
        rows = []
        for gid, decisions in games:
            f = factory(gid)
            if f is None:
                continue
            r = score(f, [(gid, decisions)], args.granularity)
            rows.append((gid[:4], r))
        tot = sum(r["decisions"] for _, r in rows)
        cov = sum(r["coverage"] * r["decisions"] for _, r in rows) / tot
        pred = sum(r["predicted"] * r["decisions"] for _, r in rows) / tot
        print(f"module {args.module} ({args.granularity}):")
        for gid, r in sorted(rows, key=lambda x: -x[1]["determ"]):
            report("  " + gid, r)
        print(f"  ALL    policy PRED {pred:5.1%}  (weighted)")
        return

    print(f"baselines on {sum(len(d) for _, d in games)} human decisions "
          f"({args.granularity} granularity):")
    for label, fn in BASELINES.items():
        report(label, score(fn, games, args.granularity))
    print("\nPREDICTED = share of all human decisions a pure state-lookup")
    print("replays correctly. The Grand-Prize architecture needs an authored")
    print("abstraction that beats every fixed rung on this number.")


if __name__ == "__main__":
    main()
