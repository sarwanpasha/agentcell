# AgentCell

An autonomous LLM agent that builds image-based morphological-profiling pipelines from a natural-language goal, together with a harness for studying agent behavior under an explicit compute budget.

The task is mechanism-of-action (MoA) prediction on the BBBC021 benchmark, scored by not-same-compound (NSC) and not-same-compound-and-batch (NSCB) 1-nearest-neighbor retrieval accuracy over 103 treatments. Given only the goal *"maximize NSC on BBBC021"*, the agent chooses a backbone, decides whether and how long to train a weakly supervised DINO model per channel, re-embeds, selects a normalization, and stops.

---

## Repository layout

```
agentcell/
├── src/
│   ├── agent/agent_full.py        # planner loop, SU budget ledger, earned-access rule
│   └── tools/
│       ├── tool_embed.py          # primitive: treatment-level features (ResNet-50, ImageNet ViT, trained ViTs)
│       ├── tool_train_ssl.py      # primitive: train one WS-DINO channel (inline or sbatch)
│       └── tool_score.py          # primitive: NSC / NSCB with plain or per-channel L2 normalization
├── slurm/                         # sbatch scripts for agent runs, single-channel training, seed retraining
├── scripts/
│   ├── preprocess_bbbc021.py      # raw TIFF -> illumination-corrected, resized images + CSVs
│   └── ablation_fixed_pipelines.py# fixed (non-agent) pipelines F1-F5
├── cloud/                         # self-contained scripts for a single-node cloud GPU machine (SLURM setup,
│                                  # data download, multi-seed studies, checkpoint scoring, Gram-matrix scoring)
├── results/                       # metric tables (CSV)
├── trajectories/                  # format of the agent's JSON decision traces
└── docs/
    ├── REPRODUCING.md             # step-by-step reproduction guide
    └── TOOLS.md                   # tool JSON contract and argument reference
```

The tools are thin wrappers over the WS-DINO training and evaluation code of Cross-Zamirski et al. (2022), which is not redistributed here (see *Dependencies*).

---

## The agent

The agent runs as a long-lived job on one GPU. At every step a planner LLM receives the goal, the history of actions and observations, the remaining service-unit (SU) budget and the list of checkpoints it has trained in the current run, and returns a single JSON action. The action space has three primitive tools:

- `embed`: compute features with a chosen backbone over one or three channels;
- `train_ssl`: train one WS-DINO channel with a chosen weak label and number of epochs (cost 0.0125 SU per epoch and channel);
- `score`: evaluate features under plain or per-channel L2 normalization.

Under the **earned-access** rule the agent can embed with trained models only if it produced them earlier in the same run. Every thought, action, observation and metric is logged to JSON.

---

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export DEEPSEEK_API_KEY=...      # DeepSeek-V3 planner (deepseek-chat)
export OPENAI_API_KEY=...        # optional, for the GPT-5.5 planner

python src/agent/agent_full.py \
  --goal "Build a BBBC021 mechanism-of-action pipeline that maximizes NSC accuracy (also report NSCB)." \
  --budget 14 --su-budget 60 --workdir ./runs/my_run
```

On a SLURM cluster use `slurm/run_agent.sbatch` (see `docs/REPRODUCING.md`). The upstream WS-DINO code, the DINO ViT-S/8 weights and the BBBC021 images must be staged first; `docs/REPRODUCING.md` lists the steps.

### Single cloud GPU machine

`cloud/` contains scripts that turn one multi-GPU cloud instance into a single-node SLURM cluster, download and preprocess BBBC021 in parallel, and run multi-seed training studies. See `cloud/README.md`.

---

## Results tables

| File | Content |
|---|---|
| `results/main_results.csv` | fixed pipelines and agent runs on the primary cluster |
| `results/seed_variance.csv` | 4-seed retraining of the agent's configuration on the primary cluster |
| `results/site_b_fixed_pipelines.csv` | fixed pipelines on the second cluster |
| `results/site_b_seed_study.csv` | 8 seeds at 100 and 3 seeds at 400 epochs per channel on the second cluster |
| `results/site_b_checkpoints.csv` | NSC of checkpoints saved during training |
| `results/site_b_single_channel.csv` | single-channel NSC, untrained and trained |

---

## Dependencies

- **WS-DINO** (Cross-Zamirski et al., 2022, arXiv:2209.07819): weakly supervised DINO training and evaluation code wrapped by the tools.
- **DINO** (Caron et al., ICCV 2021): ImageNet-pretrained ViT-S/8 initialization.
- **BBBC021** (Ljosa et al., Nat. Methods 2012; Caie et al., Mol. Cancer Ther. 2010): benchmark images from the Broad Bioimage Benchmark Collection.

---

## License

Code is released under the MIT License (see `LICENSE`). Result tables are released under CC BY 4.0. WS-DINO and BBBC021 retain their own licenses and are not redistributed here.
