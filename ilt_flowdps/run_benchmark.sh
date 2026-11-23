#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$SCRIPT_DIR/src:$REPO_ROOT/Hopkins:$REPO_ROOT/Hopkins_surrogate:$REPO_ROOT/ILT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# Use an already activated Python >=3.10 environment. Every CLI option is forwarded.
exec "${PYTHON:-python}" -u -m ilt_flowdps.benchmark.cli "$@"
