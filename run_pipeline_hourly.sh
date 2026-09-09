#!/usr/bin/env bash
# Runs pipeline.py on the most recently *complete* hour-folder found on disk
# for Shadowgraph_40297765. Folder names are in the instrument's own timezone
# (not this machine's), so we can't compute the target path from the system
# clock -- instead we list what's actually there and pick from that.
# Invoked hourly by the sinker-pipeline systemd timer.
set -euo pipefail

REPO_ROOT="/opt/cfe-lab/sinker-shadowgraph-backup"
PYTHON="/opt/cfe-lab/miniconda3/envs/sinker/bin/python"
INSTRUMENT_DIR="/mnt/SINKER/MARS/Shadowgraph_40297765"
STATE_FILE="${HOME}/.local/state/sinker-pipeline/last_processed_folder"

mkdir -p "$(dirname "$STATE_FILE")"

# Hour-folders are 4 levels deep: <year>/<month>/<day>/<hour>. Zero-padded
# names sort lexicographically in chronological order.
mapfile -t FOLDERS < <(find "$INSTRUMENT_DIR" -mindepth 4 -maxdepth 4 -type d | sort)

if (( ${#FOLDERS[@]} < 2 )); then
    echo "Fewer than 2 hour-folders found under ${INSTRUMENT_DIR}; nothing complete to process yet." >&2
    exit 0
fi

# Skip the newest folder -- it may still be receiving frames -- and take the
# one before it as the latest *complete* hour.
TARGET="${FOLDERS[-2]}"

LAST_PROCESSED=""
if [ -f "$STATE_FILE" ]; then
    LAST_PROCESSED=$(cat "$STATE_FILE")
fi

if [ "$TARGET" == "$LAST_PROCESSED" ]; then
    echo "Latest complete folder (${TARGET}) already processed; nothing new. Skipping."
    exit 0
fi

echo "Processing ${TARGET}"
"$PYTHON" "${REPO_ROOT}/pipeline.py" "$TARGET"

echo "$TARGET" > "$STATE_FILE"
