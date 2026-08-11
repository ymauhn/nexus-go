# PROPELG-GO

**Protein Propagation over Embedding-Linked Graphs, for Gene Ontology**

MSc research at the Institute of Computing, UNICAMP — protein function annotation by
propagating protein language model embeddings over protein–protein interaction networks,
across the full range from parameter-free diffusion to learned graph neural networks, and
their fusion.

> Advisor: Prof. Dr. Zanoni Dias · Co-advisors: Prof. Dr. Hélio Pedrini and
> Dr. Gabriel Bianchin de Oliveira

---

## The idea

A protein's function is predictable from two complementary signals: what its sequence looks
like, and who it interacts with. Protein language models capture the first. Interaction
networks carry the second — but exploiting them usually means training a graph neural
network, which adds parameters, cost and opacity.

PROPELG-GO treats propagation as a **design axis rather than a fixed choice**:

```
      fixed  ←───────────── spectrum ─────────────→  learned
       α=1                    hybrid                 GCN · GAT · GraphSAGE
   (sequence only)         (fusion of both)        (end-to-end graph learning)
```

At one end, a non-parametric multi-hop diffusion controlled by two explicit
hyperparameters — hop count and mixing weight α — computed once, outside the training loop,
and cached. At the other, learned message passing. In between, early- and late-fusion
combinations of the two.

## Two generations

| | **First generation** — `ppi_only/` | **Second generation** — `ppi_v4/` |
|---|---|---|
| Node representation | row of the weighted adjacency matrix, restricted to the training basis | frozen ProtT5-XL-U50 embeddings, 3,072-d (layers 24/23/22, mean-pooled and concatenated) |
| Propagation | none — topology *is* the feature | fixed diffusion, GCN, GAT, GraphSAGE, and fusions |
| Heads | MLP, CNN1D, and image backbones over an outer-difference matrix | MLP, and a visual encoding into 3 × 32 × 32 RGB |
| Role | controlled topology-only baseline | the proposed method |

The first generation is **deliberately subtractive**: `equalize`, `dilate`, `enhance`,
`convexize` and the ProtT5 channels existed in the legacy notebook and were removed on
purpose, so that the topology-only ceiling could be measured cleanly. It is a control, not
an unfinished prototype.

## Why one repository, and why the nested `ppi_v4/ppi_v4/`

`ppi_only/compat.py` resolves `<repo root>/ppi_v4/ppi_v4/engine.py` at import time and
re-exports the metric, the loss and the training engine from it. That inversion is
deliberate and methodological, not technical debt: for the two generations to appear in the
same comparison table, they must share **bit for bit** the same `metrics.evaluate_collect`
and the same `losses.ProteinLoss`. Splitting them into separate repositories would force a
duplicate `metrics.py`, which is precisely what this work does not do.

Keep the layout as it is and everything imports without a patch.

---

## Layout

```
ppi_only/            first generation — 13 modules       (run from repo root)
configs/ppi_only/    its YAML
scripts/ppi_only/    its runner
ppi_v4/                                                  (run from here)
├── ppi_v4/          second generation + shared metric/loss/engine library
├── configs/         01..06 YAML
└── scripts_v4/      batch runners
tools/               report consolidation and notebook generation
notebooks/           legacy qualifying-exam notebook + the two Colab runners
figures/             architecture diagrams (SVG)
results/
├── ppi_only/        per-run metrics, ablation summaries, loss curves
├── ppi_v4/          hop ablation
└── chapter6/        consolidated tables and the analysis behind them
docs/                per-generation documentation, environment, data provenance
data/                empty in git — see docs/data.md
```

## Running

The environment, the exact package pins and the hardware used for each campaign are in
[`docs/environment.md`](docs/environment.md). Data provenance and download instructions are
in [`docs/data.md`](docs/data.md).

**The two generations run from different working directories** — this is not incidental,
it is what lets `ppi_only/compat.py` resolve the shared metric and loss from `ppi_v4`.

```bash
# first generation — from the repository root
python -m ppi_only.main --config configs/ppi_only/mock_mlp.yaml --sanity-only
python -m ppi_only.run_ablations --domains bp,cc,mf

# second generation — from inside ppi_v4/
cd ppi_v4
python -m ppi_v4.main_v32 --config configs/02_fixo_mlp.yaml --data-dir data --ppi-csv data/ppi.csv
bash scripts_v4/run_batch.sh
```

Both expect their data alongside; see [`docs/data.md`](docs/data.md) for the symlinks.

### Verified from a clean clone

Both generations were run from a fresh `git clone` of this repository against the real
corpus, and reproduce identically:

| | Command | Result |
|---|---|---|
| First generation | `python -m ppi_only.main --config configs/ppi_only/mock_mlp.yaml --sanity-only` | all six sanity gates pass |
| First generation | `... --smoke --limit-train 400 --limit-val 120` | trains, checkpoints, reports all six metrics |
| Second generation | `cd ppi_v4 && python -m ppi_v4.main_v32 --config configs/02_fixo_mlp.yaml` | trains and evaluates; on `mf`, one epoch gives fmax 0.6505, wFmax 0.5055, Smin 8.386 |

The second-generation figures are identical to the same run in the source tree, to the last
digit — the repository carries everything the pipeline needs.

### Hardware

The first generation runs comfortably on a 4 GB GPU. The second does not: building the
fixed-propagation buffer over the full node set materialises several copies of an
`n × 3072` float matrix, which needs **8 GB of VRAM** at the documented batch sizes — the
constraint the design was built around, and the reason models are never allowed to coexist
in memory (`release()` in the local VLM path, staged loading elsewhere). On a smaller card
the run fails at `models_v32.py`, in `H_fixed = mp(H0)`, with a CUDA out-of-memory error.
Forcing CPU with `CUDA_VISIBLE_DEVICES=""` works and is useful for verifying a setup, but
is far too slow for a real campaign.

## Results

[`results/chapter6/00_INDICE.md`](results/chapter6/00_INDICE.md) maps every run to the
table it appears in. The consolidated CSVs live in `results/chapter6/dados/`.

Headline test-set figures (wFmax, the primary criterion):

| Ontology | PROPELG-GO | Best baseline |
|---|--:|--:|
| Biological Process | **58.69** | 48.92 (DeepGraphGO) |
| Cellular Component | **68.65** | 60.35 (DeepGraphGO) |
| Molecular Function | **73.89** | 68.96 (DeepGraphGO) |

Full metric suite — Fmax, Fmax\*, IC-weighted Fmax, Smin, AUPRC and interpolated AUPRC — in
the chapter 6 tables.

---

## Known issues in the staged code

Listed rather than fixed: this is research code and is preserved exactly as it was run, so
that every number in the tables remains traceable to the code that produced it.

| File | Issue |
|---|---|
| `scripts/ppi_only/run_in_env.sh` | Hardcodes the conda profile path and `cd`s to an absolute project directory from the original machine. Set your own paths before use. |
| `ppi_v4/scripts_v4/run_all_experiments.sh` | Invokes `ppi_v31.main_v32`, a module that does not exist under this name, and passes `--raw-dir`, which the argument parser does not accept. Use `scripts_v4/run_batch.sh` or call `python -m ppi_v4.main_v32` directly. |
| `ppi_v4/configs/03_gnn_vision_e2e.yaml` | Requests the backbone `convnext_base`, which `ppi_v4/models.py` does not build. |
| `ppi_v4/ppi_v4/main_v32.py` | The embedding directory is fixed as `raw/raw_{domain}` relative to the working directory — there is no `--raw-dir` flag. Symlink `ppi_v4/raw` at your cache. |
| `tools/build_nb.py` | Writes its generated notebook to an absolute path from the original machine. Edit the destination before running. |

These are recorded here rather than patched. Two of them (the working directory for each
generation, and where `raw/` must live) are covered by the symlink arrangement in
[`docs/data.md`](docs/data.md) and need no code change at all.
| `results/ppi_only/` | Carries the per-run `*_metrics.csv` but not the matching `*_manifest.json`, so `python -m ppi_only.consolidate` cannot run here. Use `tools/combine_report.py` instead. |

## Licence

**All rights reserved** while the extended article is under review — see
[`NOTICE.md`](NOTICE.md). An open-source licence will be applied on publication.

## Publications

**MAUHNOOM, Y.**; OLIVEIRA, G. B.; PEDRINI, H.; DIAS, Z. *From Graphs to Images:
Non-Parametric PPI Context Integration for Vision-Based Protein Function Prediction.*
VISAPP 2026, Marbella, Spain.

**MAUHNOOM, Y.**; OLIVEIRA, G. B.; PEDRINI, H.; DIAS, Z. *Beyond Learnable Graphs: Fixed
Multi-Hop PPI Diffusion of Sequence Embeddings for Robust Protein Function Prediction.*
Extended journal version, under review.
