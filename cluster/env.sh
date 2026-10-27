export PYTHONNOUSERSITE=1
export W=$HOME/agentcell_work
source ${VENV:-$W/venv}/bin/activate
export WSDINO_REPO=$W/WS-DINO
export AGENTCELL_HOME=$W/agentcell
export AGENTCELL_TOOLS=$W/agentcell/src/tools
export AGENTCELL_DATA=${DATA_OVERRIDE:-$W/data/bbbc021_wsdino}
export AGENTCELL_RUNS=$W/runs
