#!/bin/zsh
# Push Franzen-chassis demo arms as GPU slots free up (Kaggle: 2 concurrent GPU sessions),
# pull each finished arm's output, and print its read against the base band.
#
#   nohup scripts/arm_queue.sh census drain80 temp06 > scratchpad/arm-queue.out 2>&1 &
#
# Arms already queued/running are discovered from the kernel list below; add names there
# when a new arm is pushed by hand.
cd "$(dirname "$0")/.."
LOG=scratchpad/arm-queue.log; say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
bounded() { perl -e 'alarm 900; exec @ARGV' "$@"; }
KNOWN=(yield4096 consolidate "$@")
PULLED=()
status_of() { bounded kaggle kernels status "tantan0327/arc3-franzen-$1" 2>&1 | tail -1; }
running() { local n=0; for k in "${KNOWN[@]}"; do st=$(status_of "$k"); case "$st" in *RUNNING*|*QUEUED*) n=$((n+1));; esac; done; echo $n; }
pull_finished() {
  for k in "${KNOWN[@]}"; do
    [[ " ${PULLED[*]} " == *" $k "* ]] && continue
    st=$(status_of "$k")
    case "$st" in
      *COMPLETE*|*ERROR*|*CANCEL*)
        say "$k finished: $st"; OUT=scratchpad/arc3-franzen-${k}_out; rm -rf "$OUT"; mkdir -p "$OUT"
        bounded kaggle kernels output "tantan0327/arc3-franzen-$k" -p "$OUT" >> "$LOG" 2>&1
        python3 scripts/read_arm.py "$OUT" 2>&1 | tee -a "$LOG"
        grep -h -o "CENSUS game=[^ ]* level=[0-9]* probes=[0-9]* executed=[0-9]* stopped=[^ ]*" "$OUT"/*.log 2>/dev/null | sort | uniq -c | head -40 >> "$LOG"
        PULLED+=("$k") ;;
    esac
  done
}
for name in "$@"; do
  while [ "$(running)" -ge 2 ]; do pull_finished; sleep 180; done
  say "pushing arm $name"
  out=$(bounded kaggle kernels push -p "notebooks-franzen-$name" 2>&1 | tail -1); say "$out"
  case "$out" in *"Maximum batch"*) say "slot race lost; retry in 3 min"; sleep 180; out=$(bounded kaggle kernels push -p "notebooks-franzen-$name" 2>&1 | tail -1); say "$out";; esac
  sleep 90
done
while [ "$(running)" -gt 0 ]; do pull_finished; sleep 180; done
pull_finished
say "queue drained"
