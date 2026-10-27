#!/bin/bash
#SBATCH -J agentCell_setup_extra
#SBATCH -c 60
#SBATCH --mem=200G
# Second preprocessing recipe (unit-mean illumination) and a second environment with PyTorch 2.5.1.
set -euo pipefail
W=$HOME/agentcell_work; S=$W/setup_files
(
  export PYTHONNOUSERSITE=1
  [ -d $W/venv251 ] || python3 -m venv --system-site-packages $W/venv251
  source $W/venv251/bin/activate
  pip install -q "numpy<2" "albumentations==1.3.1" "opencv-python-headless<4.11" tifffile timm scikit-image h5py "pyarrow<18" "scikit-learn==1.5.2" "pandas<2.3"
  pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  python -c "import torch,torchvision;print('VENV251',torch.__version__,torchvision.__version__,torch.cuda.is_available())"
) > $W/logs/venv251.log 2>&1 &
P=$!
source $S/env.sh
python $S/preprocess_unitmean.py --raw-dir $W/raw --wsdino-repo $W/WS-DINO --out-dir $W/data/bbbc021_wsdino_unitmean --workers 56
wait $P || echo "venv251 setup failed"; tail -2 $W/logs/venv251.log
echo "EXTRA_DONE $(date)"
