#!/usr/bin/env bash
# Reproduction kit for a published Kaggle kernel (docs/publication-read-protocol.md §1-§3):
# pull it byte-for-byte with its metadata, list every attached source with its licence,
# check the image digest and the GPU shape, and stage a kernel dir under our account.
#
#   scripts/repro_kit.sh <owner>/<kernel-slug> <our-name>      # e.g. scripts/repro_kit.sh da-fr/arc3-final arc3-repro-franzen
#
# Prints a checklist; does NOT push. Push with: kaggle kernels push -p notebooks-<our-name>
set -uo pipefail
cd "$(dirname "$0")/.."
REF=${1:?kernel ref}; NAME=${2:?our kernel name}; SLUG=${REF##*/}
S=scratchpad/repro/$SLUG; rm -rf "$S"; mkdir -p "$S"
echo "== pull $REF"; kaggle kernels pull "$REF" -m -p "$S" 2>&1 | tail -2
META="$S/kernel-metadata.json"; [ -f "$META" ] || { echo "no kernel-metadata.json (private or not a notebook?)"; exit 1; }
CODE=$(python3 -c "import json;print(json.load(open('$META'))['code_file'])")
echo "== notebook: $S/$CODE  sha256 $(shasum -a 256 "$S/$CODE" | cut -c1-16)  cells $(python3 -c "import json;print(len(json.load(open('$S/$CODE')).get('cells',[])))" 2>/dev/null)"
python3 - "$META" <<'PY'
import json,sys,subprocess
m=json.load(open(sys.argv[1]))
print("== metadata: gpu",m.get("enable_gpu"),"| internet",m.get("enable_internet"),"| shape",m.get("machine_shape"),"| image",(m.get("docker_image") or "")[-24:])
U=json.load(open(__import__("os").path.expanduser("~/.kaggle/kaggle.json"))); auth=U["username"]+":"+U["key"]
for ds in m.get("dataset_sources",[]):
    out=subprocess.run(["curl","-s","-u",auth,f"https://www.kaggle.com/api/v1/datasets/view/{ds}"],capture_output=True,text=True).stdout
    try: d=json.loads(out); print(f"   dataset {ds}: licence={d.get('licenseName')} bytes={d.get('totalBytes')} private={d.get('isPrivate')}")
    except Exception: print(f"   dataset {ds}: (no metadata: {out[:80]})")
for ms in m.get("model_sources",[]):
    parts=ms.split("/"); inst="/".join(parts[:4])
    out=subprocess.run(["kaggle","models","instances","get",inst],capture_output=True,text=True).stdout
    lic=[l for l in out.splitlines() if "licen" in l.lower()]
    print(f"   model {ms}: {' '.join(x.strip() for x in lic)[:120] or '(licence not shown; open the model page)'}")
for ks in m.get("kernel_sources",[]): print("   kernel source (a dependency notebook, check separately):",ks)
PY
echo "== grep for private/unattached inputs referenced in the code"
python3 - "$S/$CODE" <<'PY'
import json,re,sys
src="\n".join("".join(c.get("source",[])) for c in json.load(open(sys.argv[1])).get("cells",[]) if c.get("cell_type")=="code")
paths=sorted(set(re.findall(r"/kaggle/input/[A-Za-z0-9_./-]+",src)))
print("   /kaggle/input paths referenced:",len(paths)); [print("    ",p) for p in paths[:20]]
for pat in ("http://","https://","requests.","urllib","subprocess","pip install","kaggle.com"): print(f"   {pat:14s} {src.count(pat)}")
PY
D=notebooks-$NAME; mkdir -p "$D"; cp "$S/$CODE" "$D/duck.ipynb"
python3 - "$META" "$D" "$NAME" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); m["id"]=f"tantan0327/{sys.argv[3]}"; m["title"]=f"ARC3 {sys.argv[3]}"; m["code_file"]="duck.ipynb"; m["is_private"]=True
# Only the standard fields: the pulled metadata carries the source kernel's `id_no`, and pushing
# it makes the API try to set that kernel's IAM policy -> 403 "kernels.setIamPolicy" (10/1).
KEEP=("id","title","code_file","language","kernel_type","is_private","enable_gpu","enable_tpu","enable_internet","dataset_sources","kernel_sources","competition_sources","model_sources","docker_image","machine_shape","keywords")
m={k:m[k] for k in KEEP if k in m}
json.dump(m,open(sys.argv[2]+"/kernel-metadata.json","w"),indent=2); print("== staged",sys.argv[2],"->",m["id"],"(byte-identical notebook; push only after the licence checklist passes)")
PY
