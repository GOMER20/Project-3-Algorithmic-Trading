#!/bin/zsh
# Starts AT for the next US market session in the Gomer Trading app on this Mac.
# AT normally starts by itself (see README, "Automatic sessions"); this is for a session by
# hand. Double-click it any time before the open; it waits, trades the session, then reports.
# If AT is already running it says so and does nothing.
cd "$(dirname "$0")" || exit 1
mkdir -p runs
echo "AT: waiting for the US market session. Leave this window open."
.venv/bin/python scripts/run_at.py --us-session --once 2>&1 | tee -a runs/at-session.log
echo
echo "AT has finished. The report is in runs/latest.md. You can close this window."
