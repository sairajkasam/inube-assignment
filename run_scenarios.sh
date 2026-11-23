#!/usr/bin/env bash
# Reproduces traces/s1..s4. S2 and S3 start from the state S1 saved.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export TRACCIA_SEED=${TRACCIA_SEED:-13}
MODEL=${TRACCIA_MODEL:-gemini-3-flash-preview}

$PY agent.py --model "$MODEL" --scenario S1 --profile data/profile.json \
  --conversation data/conversation_C-204.json --trace traces/s1.json --save-state traces/s1_state.json
$PY agent.py --model "$MODEL" --scenario S2 --state traces/s1_state.json \
  --conversation data/conversation_C-205.json --trace traces/s2.json
$PY agent.py --model "$MODEL" --scenario S3 --state traces/s1_state.json \
  --conversation data/conversation_C-206.json --trace traces/s3.json
TRACCIA_FORCE_FAIL=1 $PY agent.py --model "$MODEL" --scenario S4 --profile data/profile.json \
  --conversation data/conversation_C-204.json --trace traces/s4.json
