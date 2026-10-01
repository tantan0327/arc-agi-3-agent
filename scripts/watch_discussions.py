"""Snapshot the competition's discussion threads through the Kaggle SDK.

Kaggle's web pages are SPAs that fetch returns as a bare title; the SDK the
CLI ships with (kagglesdk) exposes the topic list and every message tree. Used
from 9/29 to catch the leaders' publication decisions (topics 742801, 742935,
743624) and any "I published" post, and from 10/1 for the write-ups.

    python scripts/watch_discussions.py            # newest topics + watched threads, new posts only
    python scripts/watch_discussions.py --all      # print every post of the watched threads
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiListCompetitionTopicsRequest, ApiListTopicMessagesRequest, TopicListSortBy)

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "scratchpad" / "discussions" / "seen.json"
COMP = "arc-prize-2026-arc-agi-3"
WATCH = [742801, 742935, 743624, 743753, 743785, 742940, 743723]


def walk(posts, depth=0):
    for p in posts or []:
        yield depth, p
        yield from walk(p.get("replies") or [], depth + 1)


def to_dict(m):
    d = {"id": getattr(m, "id", None), "postDate": str(getattr(m, "post_date", "")), "votes": getattr(m, "votes", 0),
         "author": getattr(m, "author_name", "") or "",
         "rawMarkdown": getattr(m, "raw_markdown", None) or getattr(m, "content", "") or "",
         "replies": [to_dict(r) for r in (getattr(m, "replies", None) or [])]}
    return d


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--all", action="store_true"); ap.add_argument("--watch", type=int, action="append", default=[])
    args = ap.parse_args()
    api = KaggleClient().competitions.competition_api_client
    STATE.parent.mkdir(parents=True, exist_ok=True)
    seen = set(json.loads(STATE.read_text())) if STATE.exists() else set()
    new_ids = set()
    req = ApiListCompetitionTopicsRequest(); req.competition_name = COMP; req.sort_by = TopicListSortBy.TOPIC_LIST_SORT_BY_NEW
    topics = getattr(api.list_competition_topics(req), "topics", None) or []
    print("-- newest topics:")
    for t in topics[:20]:
        tid_ = int(getattr(t, "id", 0) or 0); tag = "NEW TOPIC " if tid_ and tid_ not in seen else ""
        print(f"  {tag}{tid_}  {str(getattr(t, 'title', ''))[:80]}  votes {getattr(t, 'votes', '')}  {str(getattr(t, 'post_date', ''))[:16]}")
        new_ids.add(tid_)
    for tid in WATCH + args.watch:
        r = ApiListTopicMessagesRequest(); r.topic_id = tid; r.competition_name = COMP   # both are required
        try:
            res = api.list_topic_messages(r)
        except Exception as exc:  # noqa: BLE001
            print(f"-- topic {tid}: {type(exc).__name__}: {str(exc)[:120]}"); continue
        msgs = [to_dict(m) for m in (getattr(res, "messages", None) or [])]
        (STATE.parent / f"topic_{tid}.json").write_text(json.dumps(msgs, indent=1))
        posts = list(walk(msgs))
        fresh = [(d, p) for d, p in posts if p["id"] not in seen]
        print(f"-- topic {tid}: {len(posts)} posts, {len(fresh)} new")
        for d, p in (posts if args.all else fresh):
            txt = re.sub(r"<[^>]+>", " ", p["rawMarkdown"]); txt = re.sub(r"\s+", " ", txt)[:500]
            print(f"  {'  ' * d}[{p['postDate'][:16]} v{p['votes']} {p['author']}] {txt}")
            new_ids.add(p["id"])
    STATE.write_text(json.dumps(sorted(seen | new_ids)))


if __name__ == "__main__":
    main()
