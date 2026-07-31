#!/bin/bash
# Pulls DriverAssist's debug/detection logs off the device into ~/DriverAssist/logs/.
#
# Archived logs (overlay-debug-<epoch>.log, detections-<epoch>.jsonl) are immutable
# once rotated (see DebugFileLogger.reset()/DetectionLogger.reset()), so they're only
# pulled if not already present locally. The "current" logs (overlay-debug.log,
# detections.jsonl) are still being written to by a running app, so they're always
# re-pulled to pick up the latest content.
#
# Run this any time — after a drive, or whenever — there's no harm in re-running it.
set -euo pipefail

DEVICE_ID="00008150-00094CE00EF2401C"
BUNDLE_ID="com.rickclark.DriverAssist"
DEST_DIR="$HOME/DriverAssist/logs"

mkdir -p "$DEST_DIR"

pull() {
    local remote_name="$1"
    xcrun devicectl device copy from \
        --device "$DEVICE_ID" \
        --domain-type appDataContainer \
        --domain-identifier "$BUNDLE_ID" \
        --source "/Documents/$remote_name" \
        --destination "$DEST_DIR/$remote_name" \
        >/dev/null 2>&1
}

echo "Listing files on device..."
remote_files=$(xcrun devicectl device info files \
    --device "$DEVICE_ID" \
    --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" 2>/dev/null \
    | grep -E "overlay-debug.*\.log|detections.*\.jsonl" \
    | awk '{print $1}' \
    | sed 's|^Documents/||')

if [ -z "$remote_files" ]; then
    echo "No log files found on device (is the app installed and has it run at least once?)"
    exit 0
fi

pulled=0
skipped=0
for name in $remote_files; do
    case "$name" in
        overlay-debug.log|detections.jsonl)
            # Current/active log — always re-pull for latest content.
            pull "$name"
            echo "  pulled (current): $name"
            pulled=$((pulled + 1))
            ;;
        *)
            # Archived log — immutable once rotated, skip if already local.
            if [ -f "$DEST_DIR/$name" ]; then
                skipped=$((skipped + 1))
            else
                pull "$name"
                echo "  pulled (new):     $name"
                pulled=$((pulled + 1))
            fi
            ;;
    esac
done

echo "Done: $pulled pulled, $skipped already up to date. Logs in $DEST_DIR"
