#!/bin/bash
# bootstrap_node.sh SPECFILE [extra] [lincs]   -- run once per node
set -euo pipefail
W=$HOME/agentcell_work; mkdir -p $W/logs $W/grams; cd $W/setup_files
N=$(( $(nproc) - 4 ))
sed -i "s/^#SBATCH -c 32$/#SBATCH -c $N/; s/--workers 30/--workers $N/" sb_setup.sbatch
sed -i "s/^#SBATCH -c 60$/#SBATCH -c $N/; s/--workers 56/--workers $N/" sb_setup_extra.sbatch
SLOTS=${SLOTS:-3} bash slurm_setup.sh > $W/logs/slurm_setup.log 2>&1
LAST=$(sbatch --parsable sb_setup.sbatch); echo "setup $LAST"
if [[ " $* " == *" extra "* ]]; then LAST=$(sbatch --parsable -d afterok:$LAST sb_setup_extra.sbatch); echo "extra $LAST"; fi
if [[ " $* " == *" lincs "* ]]; then
  mkdir -p $W/lincs/meta/platemap; cd $W/lincs/meta
  B=https://cellpainting-gallery.s3.amazonaws.com; P=cpg0004-lincs/broad/workspace/metadata
  curl -s -O $B/$P/external_metadata/external_metadata.tsv; curl -s -O $B/$P/platemaps/2016_04_01_a549_48hr_batch1/barcode_platemap.csv
  for k in $(curl -s "$B/?list-type=2&prefix=$P/platemaps/2016_04_01_a549_48hr_batch1/platemap/" | grep -o "<Key>[^<]*" | sed 's/<Key>//'); do curl -s -o platemap/$(basename $k) $B/$k; done
  cd $W/setup_files
  LAST=$(sbatch --parsable -d afterok:$LAST -c $N --mem=200G -J agentCell_lincs_prep -o $W/logs/lincs_prep_%j.out \
     --wrap "source $W/setup_files/env.sh; python $W/setup_files/lincs_prep.py --meta $W/lincs/meta --out $W/data/lincs_wsdino --workers 16")
  echo "lincs $LAST"
fi
bash submit_runs.sh $1 $LAST
