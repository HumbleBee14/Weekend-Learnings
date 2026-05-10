#!/usr/bin/env bash
# Confirm an actual Thunderbolt 5 link between two Macs before debugging
# distributed inference. Requires iperf3 on both ends.
#
# Usage:
#   on the server Mac:  iperf3 -s
#   on the client Mac:  bash tb5_link_probe.sh <server-tb-bridge-ip>
#
# Expected:
#   TB5: 60-75 Gb/s sustained
#   TB4: 25-32 Gb/s sustained
#   USB-C 3.x fallback: <10 Gb/s — your "TB5 cable" did not negotiate as TB5.

set -euo pipefail

SERVER="${1:-}"
if [[ -z "$SERVER" ]]; then
  echo "usage: $0 <server-tb-bridge-ip>" >&2
  exit 1
fi
if ! command -v iperf3 >/dev/null 2>&1; then
  echo "install iperf3:  brew install iperf3" >&2
  exit 1
fi

echo "TCP, 8 parallel streams, 20 seconds:"
iperf3 -c "$SERVER" -P 8 -t 20

echo
echo "interpret:"
echo "  >55 Gbps : real TB5 link"
echo "  25-35 Gbps: TB4 (still usable for 70B, struggles past 200B)"
echo "  <15 Gbps : USB-C fallback or wrong interface — check System Settings"
echo "             > Network and confirm the 'Thunderbolt Bridge' is up."
