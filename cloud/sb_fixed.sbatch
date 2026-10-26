#!/bin/bash
#SBATCH -J agentCell_fixed
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -o %x_%j.out
# Fixed (non-agent) pipelines F1-F5 on this preprocessing. Env: GPU
set -euo pipefail
source $HOME/agentcell_work/setup_files/env.sh
export CUDA_VISIBLE_DEVICES=$GPU
cd $WSDINO_REPO
python $AGENTCELL_HOME/scripts/ablation_fixed_pipelines.py --workdir $AGENTCELL_RUNS/fixed
