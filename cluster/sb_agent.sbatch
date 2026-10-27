#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=48G
# One agent run. Env: NAME GPU FLAGS
set -euo pipefail
source $HOME/agentcell_work/setup_files/env.sh
source $W/keys.sh
export CUDA_VISIBLE_DEVICES=$GPU
export AGENTCELL_TOOLS=$W/agentcell/src/tools AGENTCELL_TRAIN_CACHE=$W/runs/rev
FLAGS=${FLAGS//SPLITFILE/$W/split_seed0.json}
cd $WSDINO_REPO
python $W/agentcell/src/agent/agent_full.py \
  --goal "Build a BBBC021 mechanism-of-action pipeline that maximizes NSC accuracy (also report NSCB)." \
  --budget ${BUDGET:-14} --su-budget 60 --workdir $W/runs/agent/$NAME $FLAGS
