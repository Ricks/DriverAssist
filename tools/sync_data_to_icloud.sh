#!/bin/bash
# Daily one-way backup of DriverAssist/data/ into the iCloud-synced
# DriverAssistData mirror -- NOT the working copy tools read from (that
# stays local at DriverAssist/data/ so nothing depends on iCloud's
# on-demand download behavior, and no sync lag can interfere with an
# active recording/processing run).
#
# Deliberately no --delete: a backup exists to survive an accidental local
# deletion, so a file removed from the local working copy stays in the
# iCloud mirror rather than being pruned to match.
#
# Installed as a LaunchAgent (com.driverassist.icloudsync.plist) running
# daily at 03:00 -- see that plist for the schedule itself; this script is
# just the payload it runs.

set -euo pipefail

SRC="$HOME/DriverAssist/data/"
DEST="$HOME/Library/Mobile Documents/com~apple~CloudDocs/DriverAssistData/data/"
LOG="$HOME/DriverAssist/tools/sync_data_to_icloud.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting sync: $SRC -> $DEST" >> "$LOG"
rsync -av --stats "$SRC" "$DEST" >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sync complete." >> "$LOG"
