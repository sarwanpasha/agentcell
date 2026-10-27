# AgentCell

An LLM agent that builds image-based morphological profiling pipelines from a natural-language goal, together with a harness for auditing what the agent decided, how stable the result is and whether it transfers.

The task is mechanism-of-action (MoA) prediction on the BBBC021 benchmark. Each treatment (a compound at one concentration) is summarized by a feature vector, and quality is measured by 1-nearest-neighbor retrieval over 103 treatments:

* **NSC** (not-same-compound): the nearest treatment of a different compound must share the query's MoA.
* **NSCB** (not-same-compound-or-batch): as NSC, and the neighbor must also come from a different imaging batch.

Given only the goal "maximize NSC on BBBC021", the agent chooses a backbone, decides whether and how long to train a weakly supervised DINO (WS-DINO) model per channel, re-embeds, chooses a normalization and stops. Training is paid from a service-unit (SU) budget, and the agent may only embed with models it trained itself in the same run (earned access).

## Quick start (no GPU, no data, about one minute)

Everything in this section runs on a laptop with Python 3.9 or later.

```bash
git clone https://github.com/sarwanpasha/agentcell.git
cd agentcell
python -m venv venv && source venv/bin/activate
pip install numpy scipy

# 1. How scoring works, on synthetic data
python examples/toy_example.py

# 2. Recompute the statistics of the study from the released per-treatment outcomes
python analysis/analysis_compact.py results/compact_outcomes.json results/report.json
python analysis/make_tables.py results/report.json results/tables
```

Expected output of step 1:

```
36 treatments, 18 compounds, 6 MoAs, 4 batches
plain  NSC 0.972  NSCB 0.972  (Gram scoring identical)
l2     NSC 0.972  NSCB 0.972  (Gram scoring identical)
single channel DNA      NSC 0.944
single channel tubulin  NSC 1.000
single channel actin    NSC 0.417
per-channel L2 on the selection half 0.958, on the test half 1.000
```

The toy example builds treatments with compounds, MoAs, batches and three channels, scores them with NSC and NSCB under plain concatenation and per-channel L2 normalization, verifies that scoring from per-channel Gram matrices is exact, and evaluates a configuration on held-out compounds.

Expected output of step 2 (the last command):

```
untrained ViT, plain:        NSC 0.9126  NSCB 0.7379
untrained ViT, per-ch. L2:   NSC 0.9223  NSCB 0.6893
compound 100 ep, L2, final:  NSC 0.8835  epoch 1: 0.9709
held-out test NSC  F2 0.916  agent config 0.885  grid 0.934 (without MoA label 0.884)
checkpoint selection (compound 100, L2): test NSC 0.954 vs final 0.890
LINCS untrained ViT plain NSC 0.5000, trained seed 2 epoch 100 0.5278
tables written to results/tables
```

`results/compact_outcomes.json` stores, for every scored configuration and saved checkpoint, whether each treatment's nearest admissible neighbor shares its MoA (`1`), does not (`0`), or does not exist (`-`), under both normalizations and both rules, plus single-channel outcomes and all mixes of trained and untrained channel encoders. From these strings `analysis_compact.py` computes bootstrap intervals, exact McNemar tests, held-out grid and random search over 1,000 compound splits, and checkpoint selection on held-out compounds. `make_tables.py` writes the CSV tables in `results/tables/`.

## How scoring works

For channel c, let `G_c = X_c X_c^T` be the Gram matrix of the treatment profiles after typical variation normalization (TVN). Then

* plain concatenation: `sim = (sum_c G_c)` divided by the outer product of the square roots of its diagonal,
* per-channel L2 normalization: `sim = mean_c G_c / sqrt(diag(G_c) diag(G_c)^T)`, the mean of the per-channel cosines.

So three 103 x 103 matrices are enough to score any combination of channel encoders, including channels trained on different machines. `analysis/embed_ckpts.py` writes these matrices for trained checkpoints, and `analysis/export_compact.py` turns a folder of them into the compact outcome file.

## Repository layout

```
agentcell/
├── src/
│   ├── agent/agent_full.py      planner loop, SU ledger, earned access, held-out split option,
│   │                            checkpoint reporting, hint ablation, token logging
│   └── tools/
│       ├── tool_embed.py        treatment profiles (ResNet-50, ImageNet ViT, trained ViTs);
│       │                        exact checkpoint resolution and strict weight loading
│       ├── tool_train_ssl.py    train one WS-DINO channel; optional snapshots and cache
│       └── tool_score.py        NSC / NSCB, plain or per-channel L2, optional query subset
├── analysis/
│   ├── embed_ckpts.py           embed final models and snapshots, write per-channel Gram matrices
│   ├── export_compact.py        Gram folder -> compact per-treatment outcome file
│   ├── analysis_compact.py      statistics from the compact file (runs without a GPU)
│   ├── analysis.py              the same statistics directly from Gram matrices
│   ├── make_tables.py           CSV tables and summary
│   ├── make_split.py            selection / test split of compounds, stratified by MoA
│   └── peek.py                  print NSC / NSCB of every checkpoint in a Gram folder
├── cluster/                     scripts for one multi-GPU node (training, embedding and agent queues)
│   └── specs/                   run specifications used in the study
├── lincs/lincs_prep.py          builds the LINCS Cell Painting (cpg0004, A549) MoA benchmark
├── cloud/, slurm/, scripts/     data download, preprocessing and SLURM helpers
├── examples/toy_example.py      self-contained scoring example
├── results/                     compact outcomes, report and CSV tables
└── docs/                        reproduction guide and tool reference
```

## Full reproduction on GPUs

The tools wrap the WS-DINO code of Cross-Zamirski et al. (2022), which is not redistributed here. A full run needs one or more NVIDIA GPUs with at least 24 GB of memory each (one 100-epoch channel takes about 75 minutes on an A100).

1. Install the training environment (PyTorch 2.5.1 and 2.7 gave identical scores):

   ```bash
   pip install -r requirements.txt
   ```

2. Stage WS-DINO, the DINO ViT-S/8 ImageNet weights and BBBC021, and preprocess the images (see `docs/REPRODUCING.md` and `cloud/README.md`). All scripts read their locations from `cluster/env.sh`, which uses `$HOME/agentcell_work` as the working folder, with this repository cloned to `$HOME/agentcell_work/agentcell` and the job scripts copied next to it:

   ```bash
   mkdir -p $HOME/agentcell_work/setup_files $HOME/agentcell_work/logs
   cp cluster/*.sh cluster/*.sbatch cluster/*.py analysis/embed_ckpts.py lincs/lincs_prep.py $HOME/agentcell_work/setup_files/
   ```

3. Train and embed the configurations of a specification file, one line per run (`NAME WEAK_LABEL EPOCHS SEED SNAPSHOT_EVERY VENV PREPROCESSING`):

   ```bash
   python cluster/gpu_runner.py cluster/specs/training_a100.txt --per-gpu 1
   ```

   Each finished channel is embedded automatically, and Gram matrices appear under `$HOME/agentcell_work/grams/<run>/`.

4. Score and export:

   ```bash
   python analysis/peek.py $HOME/agentcell_work/grams c100_s0
   python analysis/export_compact.py $HOME/agentcell_work/grams my_outcomes.json
   python analysis/analysis_compact.py my_outcomes.json my_report.json
   ```

5. Run agents. Put your planner keys in `$HOME/agentcell_work/keys.sh` (for example `export DEEPSEEK_API_KEY=...` and `export OPENAI_API_KEY=...`; this file is never committed), then

   ```bash
   python cluster/agent_runner.py cluster/specs/agents.txt --gpus 0 --per-gpu 2
   ```

   Each line of `agents.txt` is `NAME|FLAGS`. Useful flags of `src/agent/agent_full.py`: `--no-hints` removes the context block of the prompt, `--split-file` restricts `score` to the selection half and scores the final pipeline once on the test half, and `--report-checkpoints` makes training return snapshot scores. Every step is logged to `runs/agent/<NAME>/`, and `final.json` holds the declared pipeline, its scores and the planner token usage.

## Second dataset

`lincs/lincs_prep.py` builds a benchmark with the same file layout from the LINCS Cell Painting collection (A549 cells, 12 MoAs, 72 compounds, 144 treatments). DNA, ER and AGP take the places of DNA, tubulin and actin, and the plate layout takes the place of the batch for NSCB. Images are read from the public Cell Painting Gallery bucket.

## Results tables

| File | Content |
| --- | --- |
| `results/compact_outcomes.json` | per-treatment outcomes of every scored configuration and checkpoint |
| `results/report.json` | output of `analysis_compact.py` |
| `results/tables/grid_configurations.csv` | the 18 grid configurations with cost, NSC and NSCB |
| `results/tables/heldout_strategies.csv` | test-half NSC of selection strategies over 1,000 splits |
| `results/tables/checkpoint_selection.csv` | checkpoint selection on held-out compounds |
| `results/tables/checkpoint_scores.csv` | NSC / NSCB of every saved checkpoint |
| `results/tables/gh200_checkpoint_scores.csv` | the same for the runs on the GH200 node |
| `results/tables/mixed_sources.csv` | all mixes of trained and untrained channel encoders |
| `results/tables/mcnemar_tests.csv` | exact McNemar tests between paired configurations |
| `results/tables/single_channel.csv` | single-channel NSC |
| `results/site_b_*.csv` | seed, checkpoint, fixed-pipeline and single-channel tables of the first replication |

Run names: `c`, `t` and `m` stand for the compound, treatment and MoA weak labels, followed by epochs per channel and seed (`c100_s0`); `um` marks the unit-mean preprocessing recipe, `f_251` PyTorch 2.5.1, `gh_` the GH200 node and `L_` the LINCS benchmark.

## Dependencies

* WS-DINO (Cross-Zamirski et al., 2022, arXiv:2209.07819): weakly supervised DINO training and evaluation code wrapped by the tools.
* DINO (Caron et al., ICCV 2021): ImageNet-pretrained ViT-S/8 initialization.
* BBBC021 (Ljosa et al., Nat. Methods 2012; Caie et al., Mol. Cancer Ther. 2010), from the Broad Bioimage Benchmark Collection.
* LINCS Cell Painting (Way et al., Mol. Biol. Cell 2022), from the Cell Painting Gallery (cpg0004).

## License

Code is released under the MIT License (see `LICENSE`). Result tables are released under CC BY 4.0. WS-DINO, BBBC021 and the LINCS images retain their own licenses and are not redistributed here.
