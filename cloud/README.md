# Cloud scripts

Scripts for running the training and scoring studies on a single multi-GPU cloud instance. Copy the contents of this folder to `$HOME/agentcell_work/setup_files/` on the instance; all scripts assume that location and write under `$HOME/agentcell_work/`.

## Setup

| Script | Purpose |
|---|---|
| `slurm_setup.sh` | Installs SLURM and munge and configures a one-node cluster. Each GPU `i` is exposed as a count-only gres `g<i>` with `SLOTS` slots, so several trainings can share one GPU while staying pinned to it. |
| `sb_setup.sbatch` | Clones agentcell, WS-DINO and DINO, patches WS-DINO (`patch_wsdino.py`), builds a virtual environment, downloads the DINO ViT-S/8 weights (`make_init_ckpt.py`), downloads the BBBC021 plates in parallel and preprocesses them (`preprocess_bbbc021.py`). |
| `env.sh` | Environment variables used by all jobs. |
| `bootstrap.sh "EPOCHS:SEED ..." [fixed]` | Runs the two steps above and submits a study; `fixed` also scores the fixed pipelines. |
| `bootstrap_one.sh SEED CH EPOCHS` | For a one-GPU instance that trains and embeds a single channel of a seed. |

## Studies

| Script | Purpose |
|---|---|
| `submit_study.sh "EPOCHS:SEED ..." [DEP_JOB]` | Submits the three channel trainings of each seed round-robin over GPUs, followed by a scoring job. |
| `sb_train_channel.sbatch` | Trains one WS-DINO channel with the compound weak label and checks that the final checkpoint has the requested epoch. |
| `sb_score_seed.sbatch` | Embeds a seed with links to the three **final** checkpoints only, then scores plain and per-channel L2. |
| `sb_fixed.sbatch` | Fixed pipelines F1 to F5. |
| `sb_train_embed.sbatch`, `sb_embed_only.sbatch`, `embed_one.py` | Train and/or embed one channel, for seeds whose channels run on different machines. |
| `gram_score.py` | Scores such seeds exactly by exchanging 103 x 103 per-channel Gram matrices instead of feature files. |
| `resubmit_failed.sh "EPOCHS:SEED ..."` | Resubmits trainings without a `DONE` marker and re-chains scoring. |
| `preprocess_unitmean.py`, `preprocess_ab_test.sh` | Alternative illumination recipe (unit-mean rescaling, bicubic resampling) and an A/B test that retrains one seed on it. |
| `collect.py` | Tabulates per-seed results and mean and standard deviation per epoch budget. |

All job names start with `agentCell_`.

## Checkpoint resolution

Training directories also contain `pretrain_full_checkpoint.pth` and epoch snapshots. `sb_score_seed.sbatch` therefore builds a directory of symbolic links to the final checkpoints and prints the epoch stored in each before embedding, so the embedding tool cannot load a different file.
