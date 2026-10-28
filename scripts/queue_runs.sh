#!/usr/bin/env bash
# Run the training jobs of a list file, at most SLOTS at once, in the order of the file.
#
#   tmux new-session -d -s wisenn-queue 'scripts/queue_runs.sh scripts/queue_2026-10-29.txt 4'
#
# Each line of the list: <job name> <W&B exp tag> <arguments of python -m iwildcam.train>.
# Every job logs to W&B (tags exp:<tag>, owner:william) and prints into logs/queue/<job>.out;
# train.py also writes logs/<run_name>.log. The queue adds a row to the "Runs in progress"
# table of the group README when it starts and deletes that row when it ends.
set -u
cd "$(dirname "$0")/.."
LIST=$1
SLOTS=${2:-4}
PY=${PY:-/zfs/ai4good/groups/camera-traps/groupB/ai4good2026-cameratraps/.venv/bin/python}
README=${README:-/zfs/ai4good/groups/camera-traps/groupB/README.md}
CKPT=${CKPT:-/zfs/ai4good/groups/camera-traps/groupB/wisenn/checkpoints}
mkdir -p logs/queue
STATUS=logs/queue/queue.log
ROW="| $(date '+%Y-%m-%d %H:%M') | William | $(grep -cv '^\s*\(#\|$\)' "$LIST") runs of $(basename "$LIST"), at most $SLOTS at once (tmux wisenn-queue) | running |"

readme() {   # readme add|del: the one row of this queue, nothing else
  "$PY" - "$README" "$ROW" "$1" <<'EOF'
import sys
path, row, what = sys.argv[1:]
text = open(path).read()
head = "| started | who | what | status |\n|---|---|---|---|\n"
if what == "add" and row not in text:
    text = text.replace(head, head + row + "\n", 1)
elif what == "del":
    text = text.replace(row + "\n", "")
open(path, "w").write(text)
EOF
}

readme add
echo "$(date '+%F %T') queue start: $LIST, $SLOTS slots" >> "$STATUS"
while read -r name tag args; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  while (( $(jobs -rp | wc -l) >= SLOTS )); do wait -n; done
  echo "$(date '+%F %T') start $name" >> "$STATUS"
  (
    # shellcheck disable=SC2086  # $args is a list of arguments
    WANDB_ENTITY=ai4good-group-b WANDB_TAGS="exp:$tag,owner:william" PYTHONPATH=$PWD \
      "$PY" -m iwildcam.train $args --wandb --ckpt-dir "$CKPT" \
      > "logs/queue/$name.out" 2>&1 < /dev/null
    echo "$(date '+%F %T') end $name exit $?" >> "$STATUS"
  ) &
  sleep 20      # start the data loaders of two jobs at different times
done < "$LIST"
wait
echo "$(date '+%F %T') queue end" >> "$STATUS"
readme del
