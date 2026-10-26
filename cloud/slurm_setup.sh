#!/bin/bash
# Single-node SLURM. Each GPU i is exposed as a count-only gres "g<i>" with SLOTS slots,
# so several WS-DINO trainings share one GPU while staying pinned to it.
set -euo pipefail
SLOTS=${SLOTS:-3}
sudo apt-get update -q >/dev/null
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q slurm-wlm munge >/dev/null
HOST=$(hostname -s); CPUS=$(nproc)
MEM=$(( $(free -m | awk '/Mem:/{print $2}') - 8192 ))
NGPU=$(nvidia-smi -L | wc -l)
TYPES=""; GRES=""
for ((i=0;i<NGPU;i++)); do TYPES+="g$i,"; GRES+="g$i:$SLOTS,"; done
TYPES=${TYPES%,}; GRES=${GRES%,}
sudo mkdir -p /var/spool/slurmd /var/spool/slurmctld /var/log/slurm
sudo chown slurm:slurm /var/spool/slurmctld /var/log/slurm || true
: | sudo tee /etc/slurm/gres.conf >/dev/null
for ((i=0;i<NGPU;i++)); do echo "NodeName=$HOST Name=g$i Count=$SLOTS" | sudo tee -a /etc/slurm/gres.conf >/dev/null; done
sudo tee /etc/slurm/slurm.conf >/dev/null <<C
ClusterName=agentcell
SlurmctldHost=$HOST
MpiDefault=none
ProctrackType=proctrack/linuxproc
ReturnToService=2
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
SlurmdSpoolDir=/var/spool/slurmd
StateSaveLocation=/var/spool/slurmctld
SlurmUser=slurm
TaskPlugin=task/none
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
DefMemPerCPU=4000
GresTypes=$TYPES
JobAcctGatherType=jobacct_gather/none
AccountingStorageType=accounting_storage/none
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
NodeName=$HOST CPUs=$CPUS RealMemory=$MEM Gres=$GRES State=UNKNOWN
PartitionName=gh Nodes=$HOST Default=YES MaxTime=INFINITE State=UP OverSubscribe=NO
C
sudo systemctl enable --now munge
sudo systemctl restart slurmctld slurmd
sleep 3; sinfo; scontrol show node | grep -i gres
