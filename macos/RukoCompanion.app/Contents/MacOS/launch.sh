#!/bin/bash
# Ruko Companion launcher.
#
# macOS grants Bluetooth and microphone access to the *application bundle* that is
# responsible for a process, so everything that touches the robot runs under this bundle.
# With no arguments it starts the companion app. With arguments it runs a command under
# the bundle's permissions, e.g.  open -n -a RukoCompanion.app --args carle scan
#
# Output goes to ~/Library/Logs/RukoCompanion/<command>.log because `open` detaches stdio.

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"          # <root>/macos/RukoCompanion.app/Contents/MacOS
LOG_DIR="$HOME/Library/Logs/RukoCompanion"
mkdir -p "$LOG_DIR"
export PATH="$HOME/Library/Python/3.9/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
cd "$ROOT" || exit 1

name="${1:-companion}"; [ $# -ge 2 ] && name="$1-$2"
name="$(printf '%s' "$name" | tr -c 'A-Za-z0-9_-' '_')"
LOG="$LOG_DIR/$name.log"
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') start: ${*:-companion} (pid $$)"
  if [ $# -eq 0 ]; then
    uv run --project "$ROOT" python -m companion
  elif [ "$1" = "carle" ]; then
    shift
    uv run --project "$ROOT/carle" carle "$@"
  else
    "$@"
  fi
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') exit: $?"
} >> "$LOG" 2>&1
