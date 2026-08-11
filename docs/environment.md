# Execution environments

Every number reported in the dissertation was produced in one of the environments below.
Wall times measured in different environments are **not** comparable to one another; the
metrics are, because they are deterministic given the stored predictions.

Device capacity is reported in **GiB**, as read from the device. Peak memory per run is
reported in **MB**, which is the unit each run recorded in its manifest; it is the peak
allocated by the training process and not the total occupancy of the card.

## Hardware

| Environment | Accelerator | Video memory | Compute capability | Role |
|---|---|---|---|---|
| Server, shared node | NVIDIA RTX A5500 | 23.55 GiB | 8.6 | Second generation: fixed propagation with the vector head, graph convolutions, fusion |
| Server, same node | NVIDIA GTX 1080 Ti | 10.90 GiB | 6.1 | Second generation: vision head |
| Local laptop | NVIDIA RTX 5060 Laptop | 7.96 GiB | 12.0 | First generation: vector routes |
| Hosted, Google Colab | not recorded | not recorded | — | First generation: image routes; deepNF-derived configuration |
| Development machine | NVIDIA GTX 1050 Ti | 3.94 GiB | — | Adapted comparative configurations |

Local laptop: Intel Core i9-14900HX (24 cores, 32 threads), 31.6 GB RAM, Windows 11 Pro
(build 10.0.26200), NVIDIA driver 591.91.
Development machine: Linux, 15 GB RAM (~12 usable), 4 GB swap, NVMe.
Server: multi-GPU node shared with other users; CPU model and RAM were not recorded.

The Colab accelerator was not recorded by the pipeline, which writes only the device type
to the manifest. It is reported as `not recorded` rather than guessed.

### Notes that affect the numbers

- The server is shared, so every script fixes `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The default
  ordering does not follow `nvidia-smi` numbering, and on a node whose cards belong to
  different generations, selecting a device by index without fixing the ordering can
  dispatch a job to the wrong card.
- The GTX 1080 Ti has no tensor cores, so mixed precision there reduces memory consumption
  without the throughput gain it brings on the RTX A5500.
- The RTX 5060 Laptop is `sm_120` and requires binaries built for CUDA 12.8 or later.
  Distributions built against earlier CUDA versions install without error and fail only
  when the first kernel is launched. It therefore has its own environment.

## Software

Each run records its own library signature. The values below are the ones observed in the
manifests.

| Environment | Python | PyTorch | torchvision | NumPy |
|---|---|---|---|---|
| Server (`ppi-gpu` conda env) | 3.10 | 2.4 or later with CUDA (exact patch not recorded) | — | < 2 |
| Local laptop | 3.11.9 | 2.11.0+cu128 | — | 1.26.4 |
| Google Colab | 3.12.13 | 2.11.0+cu128 | — | 2.0.2 |
| Development machine | 3.10 | 2.4.0+cu124 | 0.19.0+cu124 | 2.2.6 |

Common dependencies: `transformers` (ProtT5 encoder), PyTorch Geometric, Deep Graph Library
(`torch-2.4/cu124` build, development machine only), NCBI BLAST+, SciPy, pandas,
scikit-learn, Matplotlib.

### Compatibility constraints encountered directly

Three constraints were hit during the experiments. Other package versions may also affect
installation, numerical behavior or performance; only these three were observed to break
something.

1. **PyTorch / PyTorch Geometric / DGL must match.** The sparse-tensor interface and the
   message-passing operators of the latter two are tied to a specific major version of the
   former. Mismatching them produces import-time failures, not silent numerical drift.
2. **NumPy must be earlier than 2.0.** `ppi_v4/metrics.py` computes both areas under the
   precision-recall curve with `numpy.trapz`, removed in the 2.x series.
3. **The ProtT5 checkpoint is identified by name, not by revision.** It is
   `Rostlab/prot_t5_xl_half_uniref50-enc`, loaded in `bfloat16` with frozen weights. No
   revision hash was pinned, so strict reproduction depends on that release remaining
   available in its present form. The extraction procedure itself is inherited unchanged
   from [SUPERMAGOv2](https://github.com/gabrielbianchin/SUPERMAGOv2/blob/main/src/scripts/extract.py).

To regenerate an exact pin list on a machine that still has the environment:

```bash
conda activate ppi-gpu
conda list --export > docs/env_server.txt
pip freeze > docs/pip_server.txt
python - <<'PY'
import sys, torch, numpy
print(sys.version)
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "cudnn", torch.backends.cudnn.version())
print("numpy", numpy.__version__)
PY
```

## Determinism

A single global seed, `1337`, is applied to `random`, NumPy, PyTorch on host and device,
and the cuDNN backend is placed in deterministic mode. The data loaders receive an explicit
worker-init function and generator, and the worker count is fixed at 4, because changing it
changes the batch sequence even under an identical seed.

These settings make the data ordering and the parameter initialization repeatable. They do
**not** force every CUDA operation onto a deterministic kernel: training uses automatic
mixed precision with dynamic loss scaling, and some reduction kernels are not
bit-reproducible. **Bitwise identity across runs or hardware platforms is not claimed.**

Gradient checkpointing is enabled in the convolution layers, without which the graph
attention operator does not fit the server card. The RNG state is preserved across the
recomputation.

## Hyperparameters, by generation

The two generations ran under different regimes and no table in the dissertation crosses
them.

| | First generation (`ppi_only`) | Second generation (`ppi_v4`, final campaign) |
|---|---|---|
| Optimizer | Adam | Adam |
| Learning rate | 1e-4 | 1e-4 |
| Weight decay | none | none |
| Batch size | 32 | 256 |
| Maximum epochs | 200 | 250 |
| Early-stopping patience | 25 | 20 |
| Monitored quantity | validation loss | validation loss |
| Head | MLP 512 → 256, dropout 0.5, no normalization | MLP 1024 → 512, batch norm, dropout 0.4 |
| Objective | ProteinLoss (BCE in the ablation) | ProteinLoss (BCE in the ablation) |
| Class weighting | information content | information content |
| Data-loading workers | 0 | 4 |
| Seed | 1337 | 1337 |
| Profile / image size | `out_size` 1024 (vector), 224 (image) | — |
| Edge threshold | — | 0.1 |
| Convolution | — | hidden 1024, 2 layers, dropout 0.3 between layers |
| Attention heads | — | 4 (hidden), 1 (output) |

Verified constant across all 96 runs of the final second-generation campaign
(`results/chapter6/dados/results_all_dedupmax.csv`).

## Measured cost

Second generation, 96 runs, ≈ 121.5 GPU-hours:

| Group | Runs | GPU-hours |
|---|---:|---:|
| Fixed propagation | 66 | 48.7 |
| Learned route (GCN / GAT) | 12 | 70.0 |
| Early fusion | 18 | 2.8 |

Adapted comparative configurations, all on the development machine:

| Configuration | Cost |
|---|---|
| Mashup | 10.0 h for the embedding, computed once and reused by the three ontologies |
| MELISSA | ≈ 10.1 h per ontology (≈ 30.3 h total); the augmented graph depends on that ontology's biclustering |
| deepNF | not measured |
| DeepGraphGO | ≈ 1 h to train six models and run inference; the PSI-BLAST step dominates, at ≈ 3 h per ontology and partition. Wall-clock estimates from file timestamps, not from an execution log |
| SEGT-GO | 34.6 h for the complete pipeline: six models, inference and metrics |
