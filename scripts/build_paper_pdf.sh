#!/bin/zsh
# docs/paper/draft.md -> docs/paper/draft.html -> docs/paper/draft.pdf (Chrome headless; no pandoc/latex on this Mac)
cd "$(dirname "$0")/.."
SRC=${1:-docs/paper/draft.md}; OUT=${SRC%.md}
.venv/bin/python - "$SRC" "$OUT.html" <<'PY'
import sys, markdown, re
src, out = sys.argv[1], sys.argv[2]
md = open(src).read()
md = md.split("## Notes to self, not for the paper")[0]          # never publish the notes
html = markdown.markdown(md, extensions=["tables", "fenced_code", "toc"])
css = """<style>body{font-family:Georgia,'Times New Roman',serif;max-width:46em;margin:2.5em auto;line-height:1.45;font-size:11.5pt;color:#111}
h1{font-size:20pt}h2{font-size:15pt;margin-top:1.6em}h3{font-size:12.5pt}code,pre{font-family:Menlo,monospace;font-size:9pt}pre{background:#f6f6f6;padding:.6em;overflow-x:auto}
table{border-collapse:collapse;font-size:9.5pt}td,th{border:1px solid #bbb;padding:3px 6px;vertical-align:top}blockquote{color:#444;border-left:3px solid #ccc;margin-left:0;padding-left:1em}
img{max-width:100%}</style>"""
title = re.search(r"^# (.+)$", md, re.M); title = title.group(1).strip() if title else "paper"
open(out, "w").write("<!doctype html><meta charset='utf-8'><title>" + title + "</title>" + css + html)
print("html written:", out)
PY
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/$OUT.pdf" "file://$PWD/$OUT.html" 2>/dev/null
ls -la "$OUT.pdf"
