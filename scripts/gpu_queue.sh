#!/bin/zsh
# Push kernels one at a time as GPU slots free (Kaggle: 2 concurrent), pull each on completion, read it.
#   KNOWN="dprime dprime-ht" scripts/gpu_queue.sh actrule actrule2
cd "$(dirname "$0")/.."
LOG=scratchpad/gpu-queue.log; say(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
bounded(){ perl -e 'alarm 900; exec @ARGV' "$@"; }
ALL=(${=KNOWN} "$@"); PULLED=()
st_of(){ bounded kaggle kernels status "tantan0327/arc3-franzen-$1" 2>&1 | tail -1; }
running(){ local n=0; for k in "${ALL[@]}"; do case "$(st_of $k)" in *RUNNING*|*QUEUED*) n=$((n+1));; esac; done; echo $n; }
pull_done(){ for k in "${ALL[@]}"; do [[ " ${PULLED[*]} " == *" $k "* ]] && continue; case "$(st_of $k)" in *COMPLETE*|*ERROR*|*CANCEL*)
  say "$k finished"; OUT=scratchpad/arc3-franzen-${k}_out; rm -rf "$OUT"; mkdir -p "$OUT"; bounded kaggle kernels output "tantan0327/arc3-franzen-$k" -p "$OUT" >> "$LOG" 2>&1
  [ -f "$OUT/benchmark.json" ] && python3 scripts/read_arm.py "$OUT" 2>&1 | tee -a "$LOG"
  grep -h -o "PRIORITY_FORM installed[^\"]\{0,100\}\|ACTRULE game=[^\"]\{0,60\}\|ACTRULE_PATCH[^\"]\{0,60\}" "$OUT"/*.log 2>/dev/null | tail -n 3 | tee -a "$LOG"; PULLED+=("$k");; esac; done; }
for name in "$@"; do
  while [ "$(running)" -ge 2 ]; do pull_done; sleep 180; done
  out=$(bounded kaggle kernels push -p "notebooks-franzen-$name" 2>&1 | tail -1); say "push $name: $out"
  case "$out" in *"Maximum batch"*|*rror*) sleep 240; out=$(bounded kaggle kernels push -p "notebooks-franzen-$name" 2>&1 | tail -1); say "retry $name: $out";; esac
  sleep 120
done
while [ "$(running)" -gt 0 ]; do pull_done; sleep 180; done; pull_done; say "queue drained"
