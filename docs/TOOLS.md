# Tool contract and reference

The agent composes three primitive tools. Each is a standalone command-line program that performs one action and prints a single JSON object as its last line of stdout. This uniform contract is what lets the planner treat the toolbox as one composable action space.

## Result envelope

Every tool emits the same JSON structure:

```json
{
  "tool":     "embed | train_ssl | score",
  "status":   "ok | error",
  "decision": { "...": "the arguments the tool acted on" },
  "metrics":  { "...": "numeric results, e.g. NSC, NSCB, est_su" },
  "message":  "human-readable summary or error description",
  "artifacts":["/paths/to/files/produced"]
}
```

On failure, `status` is `"error"`, `message` carries the reason, and (where available) a `traceback` field is included. The agent harness surfaces `message` (and the truncated traceback) back to the planner, which is what enables self-correction.

## `tool_embed.py`

Compute treatment-level features.

| Argument | Values | Meaning |
|---|---|---|
| `--model` | `resnet50`, `vit_imagenet`, `vit_trained` | backbone |
| `--channels` | `all_three`, `dapi_only` | which fluorescent channels |
| `--init` | path | single trained checkpoint (for `vit_trained`) |
| `--init-dir` | path | directory of per-channel trained checkpoints; each channel is embedded with its own model (the correct WS-DINO setup) |
| `--tag` | string | identifier under which features are written |
| `--features-out` | path | output directory |
| `--result-out` | path | where to write the result JSON |
| `--run-train-artifacts` | path | JSON list of checkpoints the agent has earned the right to use |
| `--allow-external` | flag | dev only: bypass the earned-access check |

Earned-access: with `vit_trained`, every checkpoint to be used must appear in the run's earned list (produced by prior `train_ssl` steps), unless `--allow-external` is set.

## `tool_train_ssl.py`

Train one WS-DINO channel.

| Argument | Values | Meaning |
|---|---|---|
| `--weak-label` | `compound`, `treatment`, `moa` | weak supervision signal |
| `--channel` | `DAPI`, `Tubulin`, `Actin` | channel to train |
| `--epochs` | int (≤ `--max-epochs`, default cap 400) | training length |
| `--inline` | flag | train synchronously in this process (used when the agent runs as its own batch job) |
| `--run-dir` | path | run output directory |
| `--ledger` | path | service-unit budget ledger (JSON) |
| `--run-su-budget` | float | cap on cumulative training SUs for the run |
| `--status` | job id | poll a previously submitted (non-inline) job |
| `--result-out` | path | where to write the result JSON |

The tool reports an estimated SU cost and updates the ledger; it refuses to start a training that would exceed `--run-su-budget`. The default weak-label-to-column mapping is compound→`Unique_Compounds`, treatment→`Unique_Treatments`, moa→`Unique_MoA`; the sampler weights are derived from the chosen column's cardinality, so all three weak labels are handled correctly.

## `tool_score.py`

Score embedded features.

| Argument | Values | Meaning |
|---|---|---|
| `--features-dir` | path | directory of aggregated per-channel feature CSVs |
| `--tag` | string | which embedding to score |
| `--normalization` | `plain`, `per_channel_l2` | concatenation normalization |
| `--result-out` | path | where to write the result JSON |

Scoring loads whatever channels are present for the tag (so single-channel configurations are first-class), aggregates to the treatment level, applies TVN whitening against DMSO controls, and reports NSC and NSCB over all 103 treatments by cosine 1-NN MoA retrieval.

## The planner prompt

The exact system prompt given to the planner LLM is reproduced in the paper (Listing 1) and is held fixed across all runs and planner models. It states the objective, enumerates these three tools and their arguments, supplies directional domain priors (e.g. that an ImageNet ViT reaches ~0.91, that training can help, that channels are complementary, that the normalization trades NSC against NSCB), and fixes the single-JSON-object response format. The priors are hints, not the solution: the prompt never specifies the winning weak label, epoch count, normalization, or per-channel combination.
