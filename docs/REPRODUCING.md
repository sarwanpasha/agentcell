# Reproducing the results

This guide reproduces every reported result. It assumes a Linux machine with at least one CUDA GPU (the primary runs used a single NVIDIA GH200 per job) and a SLURM scheduler for the long-running training jobs. Single-GPU workstations work too; only wall-clock differs.

## 0. Environment variables

The code locates its dependencies and outputs through environment variables, so nothing is hard-coded to a particular cluster:

```bash
export WSDINO_REPO=$HOME/WS-DINO            # the upstream WS-DINO checkout (see step 2)
export AGENTCELL_DATA=$PWD/data/bbbc021_wsdino   # preprocessed BBBC021 (see step 3)
export AGENTCELL_TOOLS=$PWD/src/tools       # where the tool wrappers live
export AGENTCELL_HOME=$PWD                  # repo root
export AGENTCELL_RUNS=$PWD/runs             # where run outputs/checkpoints go
# If you install Python deps to a user base, also export PYTHONUSERBASE.
```

## 1. Python environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The `albumentations==1.3.1` and `pyjanitor==0.31.0` pins matter: the upstream WS-DINO data pipeline uses the albumentations 1.x augmentation API, and newer pyjanitor releases caused a `pandas_flavor` import clash in our environment.

## 2. Obtain the WS-DINO upstream code

We wrap, but do not redistribute, the WS-DINO training/evaluation code of Cross-Zamirski et al. (2022, arXiv:2209.07819). Clone it to `$WSDINO_REPO`. You will also need the DINO ImageNet-pretrained ViT-S/8 weights (`dino_deitsmall8_pretrain.pth`) placed in that directory; convert them to the full-checkpoint format the trainer expects (a teacher/student/optimizer wrapper) — the trainer loads `pretrain_full_checkpoint.pth` from each output directory.

## 3. Stage and preprocess BBBC021

Download BBBC021 from the Broad Bioimage Benchmark Collection. Then preprocess raw TIFFs into illumination-corrected, resized, per-channel images plus the annotation CSVs the tools read:

```bash
python scripts/preprocess_bbbc021.py \
    --raw-dir /path/to/bbbc021/raw \
    --wsdino-repo $WSDINO_REPO \
    --out-dir $AGENTCELL_DATA
```

This produces `$AGENTCELL_DATA/images/`, `BBBC021_annotated_wsdino.csv` (2528 rows), and `BBBC021_DMSO_wsdino.csv` (1320 rows). A quick sanity check: the labeled evaluation set should cover 103 treatments, 38 compounds, 12 MoAs.

## 4. Validate the scoring path (cheap, no training)

Before any training, confirm the no-train baseline reproduces the expected number:

```bash
python src/tools/tool_embed.py --model vit_imagenet --channels all_three \
    --tag baseline --features-out runs/baseline/features \
    --result-out runs/baseline/embed.json
python src/tools/tool_score.py --features-dir runs/baseline/features \
    --tag baseline --normalization plain --result-out runs/baseline/score.json
```

Expected: NSC ≈ 0.913 (plain), 0.903 (per_channel_l2). If you do not see this, fix the data/preprocessing before proceeding — every later result builds on it.

## 5. Main agent run

```bash
export DEEPSEEK_API_KEY=...
sbatch --export=ALL,\
AGENT_GOAL="Build a BBBC021 mechanism-of-action pipeline that maximizes NSC accuracy (also report NSCB).",\
AGENT_WORKDIR="$AGENTCELL_RUNS/main_run",AGENT_STEPS=14,AGENT_SU=60 \
  slurm/run_agent.sbatch
```

Expected trajectory: baseline → train all three channels (compound, 100 epochs) → embed trained → score plain (0.932) → score per_channel_l2 (0.971) → stop. Runtime ≈ 4 h on one GH200.

## 6. Ablations and studies

```bash
# Robustness (temperature 0.4)
sbatch --export=ALL,AGENT_GOAL="...",AGENT_WORKDIR="$AGENTCELL_RUNS/robust_t04",\
AGENT_STEPS=14,AGENT_SU=60,AGENT_FLAGS="--temperature 0.4" slurm/run_agent.sbatch

# Self-correction ablation
sbatch --export=ALL,AGENT_GOAL="...",AGENT_WORKDIR="$AGENTCELL_RUNS/ablation_sc",\
AGENT_STEPS=14,AGENT_SU=60,AGENT_FLAGS="--no-self-correct" slurm/run_agent.sbatch

# Planner comparison (GPT-5.5)
export OPENAI_API_KEY=...
sbatch --export=ALL,AGENT_GOAL="...",AGENT_WORKDIR="$AGENTCELL_RUNS/gpt55",\
AGENT_STEPS=14,AGENT_SU=60,\
AGENT_FLAGS="--endpoint https://api.openai.com/v1/chat/completions --llm-model gpt-5.5 --api-key-env OPENAI_API_KEY" \
  slurm/run_agent.sbatch

# Fixed-pipeline baselines F1–F5 (no agent, no LLM)
python scripts/ablation_fixed_pipelines.py --workdir $AGENTCELL_RUNS/fixed

# Seed-variance study (four seeds of the winning configuration)
for S in 1 2 3; do sbatch --export=ALL,SEED=$S slurm/seed_retrain.sbatch; done
```

## 7. Scoring trained checkpoints unambiguously

When several trained variants of a channel exist (e.g. compound and MoA), point the embed tool at a directory containing only the intended per-channel checkpoints (one subdirectory per channel) via `--init-dir`, so the per-channel resolver cannot pick the wrong file. This is the procedure used for all planner and seed scoring; `cloud/sb_score_seed.sbatch` builds such a directory of links to the final checkpoints automatically.

## 8. Cloud studies

The multi-seed, checkpoint and single-channel studies on a second cluster were run with the scripts in `cloud/` (see `cloud/README.md`). Their results are in `results/site_b_*.csv`.
