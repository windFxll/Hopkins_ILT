#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${ILT_DEVICE:-cuda:0}"
RUN_DIR="${ILT_OUTPUT_DIR:-$SCRIPT_DIR/runs/proxy-tv-5000-$(date +%Y%m%d-%H%M%S)-$$}"
if [[ -e "$RUN_DIR" || -e "$RUN_DIR.log" ]]; then
  echo "Output exists; set ILT_OUTPUT_DIR to a fresh path: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$(dirname "$RUN_DIR")"
extra_args=()
case "${ILT_ALLOW_BASIS_MISMATCH:-0}" in
  1) extra_args+=(--allow-proxy-basis-mismatch) ;;
  0) ;;
  *) echo "ILT_ALLOW_BASIS_MISMATCH must be 0 or 1." >&2; exit 1 ;;
esac
echo "Running proxy-tv for exactly 5000 Adam steps on $DEVICE"
echo "Results: $RUN_DIR | Log: $RUN_DIR.log"
bash "$SCRIPT_DIR/run_benchmark.sh" \
  --methods proxy-tv --device "$DEVICE" --layout iccad2013-m1-test3 \
  --output "$RUN_DIR" --max-iterations 5000 --no-early-stop \
  --tv-weight "${ILT_TV_WEIGHT:-0.01}" \
  --binary-weight "${ILT_BINARY_WEIGHT:-0.001}" \
  --area-weight "${ILT_AREA_WEIGHT:-0.08}" \
  --seed 2026 --log-every 10 --progress-every "${ILT_REPORT_EVERY:-10}" \
  "${extra_args[@]}" 2>&1 | tee "$RUN_DIR.log"
