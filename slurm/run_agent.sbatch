#!/bin/bash
#SBATCH --job-name=agent_full
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Runs the autonomous agent as a long-lived GPU job so it can train inline
# (hours) and proceed through its full decision loop without disconnection risk.
#
# Set these via --export when submitting:
#   AGENT_GOAL      : the natural-language objective
#   AGENT_WORKDIR   : output dir for trace + features + checkpoints
#   AGENT_STEPS     : max agent steps (default 12)
#   AGENT_SU        : training SU budget for the run (default 60)
#   AGENT_FLAGS     : extra flags, e.g. "--no-self-correct" (default empty)
#   DEEPSEEK_API_KEY: must be exported (passed via --export=ALL,...)

set -euo pipefail
# Set PYTHONUSERBASE / virtualenv as appropriate for your site

GOAL=${AGENT_GOAL:?set AGENT_GOAL}
WORKDIR=${AGENT_WORKDIR:?set AGENT_WORKDIR}
STEPS=${AGENT_STEPS:-12}
SU=${AGENT_SU:-60}
FLAGS=${AGENT_FLAGS:-}

cd ${WSDINO_REPO:-$HOME/WS-DINO}

echo "=== agent run starting $(date) ==="
echo "goal: $GOAL"
echo "workdir: $WORKDIR  steps: $STEPS  su: $SU  flags: $FLAGS"

/usr/bin/python ${AGENTCELL_HOME:-$PWD}/src/agent/agent_full.py \
    --goal "$GOAL" \
    --budget "$STEPS" \
    --su-budget "$SU" \
    --workdir "$WORKDIR" \
    $FLAGS

echo "=== agent run finished $(date) ==="
