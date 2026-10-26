#!/bin/bash
#SBATCH -J agentCell_setup
#SBATCH -c 32
#SBATCH --mem=120G
#SBATCH -o %x_%j.out
# code + venv (in background) || BBBC021 download (16 parallel) -> preprocess (multicore)
set -euo pipefail
W=$HOME/agentcell_work; cd $W; mkdir -p raw zips
S=$W/setup_files
(
  set -e
  [ -d agentcell ] || git clone -q https://github.com/sarwanpasha/agentcell
  [ -d WS-DINO ] || git clone -q https://github.com/crosszamirski/WS-DINO
  [ -d dino ] || git clone -q https://github.com/facebookresearch/dino
  cp dino/utils.py WS-DINO/utils.py
  python3 $S/patch_wsdino.py WS-DINO
  cd agentcell
  sed -i 's|PY = "/usr/bin/python"|PY = sys.executable|' src/agent/agent_full.py scripts/ablation_fixed_pipelines.py
  sed -i 's|cmd = \["/usr/bin/python", |cmd = [sys.executable, |' src/tools/tool_train_ssl.py
  sed -i 's|^\( *\)/usr/bin/python |\1python |' slurm/*.sbatch
  cp slurm/train_wsdino_channel.sbatch src/tools/
  cd $W
  export PYTHONNOUSERSITE=1
  [ -d venv ] || python3 -m venv --system-site-packages venv
  source venv/bin/activate
  pip install -q "numpy<2" "albumentations==1.3.1" "opencv-python-headless<4.11" tifffile timm scikit-image h5py "pyarrow<18" "scikit-learn==1.5.2" "pandas<2.3"
  python -c "import numpy,torch,albumentations,cv2,tifffile,timm,sklearn,scipy,pandas;print('ENV OK numpy',numpy.__version__,'torch',torch.__version__,'gpus',torch.cuda.device_count())"
  [ -f WS-DINO/dino_deitsmall8_pretrain.pth ] || wget -q -O WS-DINO/dino_deitsmall8_pretrain.pth https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth
  python $S/make_init_ckpt.py WS-DINO
) > $W/logs/setup_code.log 2>&1 &
CODE_PID=$!
# plate list straight from the WS-DINO repo CSVs (fetched raw, so we don't wait on the clone)
for f in BBBC021_annotated.csv BBBC021_DMSO.csv; do
  [ -f $W/$f ] || wget -q -O $W/$f https://raw.githubusercontent.com/crosszamirski/WS-DINO/main/$f
done
PLATES=$(cut -d, -f3 $W/BBBC021_annotated.csv $W/BBBC021_DMSO.csv | grep -v Image_FileName | awk -F/ '{print $(NF-1)}' | sort -u)
echo "plates: $(echo $PLATES | wc -w)"
dl(){ p=$1; z=$W/zips/BBBC021_v1_images_$p.zip
  if [ ! -d $W/raw/$p ]; then
    for i in 1 2 3; do wget -q -c -O $z https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_$p.zip && break; sleep 5; done
    unzip -q -o $z -d $W/raw && rm -f $z && echo "got $p"
  fi; }
export -f dl; export W
echo $PLATES | tr ' ' '\n' | xargs -P 16 -I{} bash -c 'dl {}'
echo "DOWNLOAD_DONE $(ls raw | wc -l) plate dirs $(date)"
wait $CODE_PID; cat $W/logs/setup_code.log | tail -3
source $S/env.sh
python $S/preprocess_bbbc021.py --raw-dir $W/raw --wsdino-repo $W/WS-DINO --out-dir $W/data/bbbc021_wsdino --workers 30
python -c "import torch; torch.hub.list('facebookresearch/xcit:main')" > /dev/null
echo "SETUP_DONE $(date)"
