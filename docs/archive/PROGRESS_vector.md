# PPI-only — Vector-head ablations (MLP + CNN1D)

_Progress snapshot: 2026-07-26 20:55:23 · 14/21 runs with saved metrics_

**Status**
- `[10/12] (A) mf_mlp_max_bce_seed1337`
- `[11/12] (A+B) mf_mlp_max_protein_seed1337`
- `[12/12] (B) mf_mlp_raw_protein_seed1337`
- `✅ 12 ok, 0 falhas. Resumos em results/ABLATIONS_summary.csv e results/ABLATIONS_summary.md`
- `### mlp phase exit=0 ###`
- `### PHASE cnn1d (9 runs) ###`
- `[1/9] (B) bp_cnn1d_avg_protein_seed1337`
- `[2/9] (B) bp_cnn1d_max_protein_seed1337`
- `[3/9] (B) bp_cnn1d_raw_protein_seed1337`
- in progress: Epoch 019 | train/loss=6.2821 | val/loss=6.6666

Higher is better for wFmax, Fmax, Fmax*, AUPRC, iAUPRC. **Lower is better for Smin.** `loss=protein` unless the run name says `bce`. Config: seed 1337, batch 32, 200 epochs, patience 25, `num_workers=0`. Image heads (ResNet50/ConvNeXt, Ablation C) run separately on Colab.

## BP

| run | split | wFmax | Fmax | Fmax* | Smin | AUPRC | iAUPRC |
|---|---|---|---|---|---|---|---|
| mlp_max_bce | val | 0.3119 | 0.3786 | 0.3287 | 23.210 | 0.3588 | 0.3536 |
|  | test | 0.3173 | 0.3824 | 0.3340 | 23.670 | 0.3645 | 0.3592 |
| mlp_max_protein | val | 0.3467 | 0.4213 | 0.3759 | 23.324 | 0.4175 | 0.4476 |
|  | test | 0.3520 | 0.4249 | 0.3811 | 23.904 | 0.4233 | 0.4521 |
| mlp_avg_protein | val | 0.3188 | 0.3980 | 0.3459 | 24.869 | 0.3770 | 0.4116 |
|  | test | 0.3233 | 0.4011 | 0.3507 | 25.361 | 0.3825 | 0.4152 |
| mlp_raw_protein | val | 0.3637 | 0.4581 | 0.3897 | 23.144 | 0.4481 | 0.4675 |
|  | test | 0.3730 | 0.4625 | 0.3986 | 23.676 | 0.4557 | 0.4737 |
| cnn1d_max_protein | val | 0.3289 | 0.4058 | 0.3588 | 23.783 | 0.3564 | 0.4276 |
|  | test | 0.3351 | 0.4090 | 0.3658 | 24.188 | 0.3635 | 0.4327 |
| cnn1d_avg_protein | val | 0.3333 | 0.4066 | 0.3684 | 23.785 | 0.4231 | 0.4378 |
|  | test | 0.3404 | 0.4104 | 0.3729 | 24.311 | 0.4279 | 0.4417 |

**Best test wFmax:** `mlp_raw_protein` = 0.3730

## CC

| run | split | wFmax | Fmax | Fmax* | Smin | AUPRC | iAUPRC |
|---|---|---|---|---|---|---|---|
| mlp_max_bce | val | 0.4201 | 0.6140 | 0.5563 | 11.712 | 0.6356 | 0.6347 |
|  | test | 0.4242 | 0.6142 | 0.5577 | 11.860 | 0.6367 | 0.6355 |
| mlp_max_protein | val | 0.4608 | 0.6748 | 0.6310 | 11.404 | 0.6674 | 0.7271 |
|  | test | 0.4667 | 0.6754 | 0.6326 | 11.460 | 0.6689 | 0.7285 |
| mlp_avg_protein | val | 0.4355 | 0.6582 | 0.6154 | 11.962 | 0.5970 | 0.7029 |
|  | test | 0.4404 | 0.6578 | 0.6159 | 12.119 | 0.5980 | 0.7024 |
| mlp_raw_protein | val | 0.4825 | 0.6741 | 0.6345 | 11.960 | 0.6351 | 0.7208 |
|  | test | 0.4839 | 0.6738 | 0.6346 | 12.034 | 0.6383 | 0.7220 |

**Best test wFmax:** `mlp_raw_protein` = 0.4839

## MF

| run | split | wFmax | Fmax | Fmax* | Smin | AUPRC | iAUPRC |
|---|---|---|---|---|---|---|---|
| mlp_max_bce | val | 0.3785 | 0.5445 | 0.4177 | 8.980 | 0.5138 | 0.5134 |
|  | test | 0.3718 | 0.5428 | 0.4153 | 8.904 | 0.5151 | 0.5146 |
| mlp_max_protein | val | 0.4672 | 0.6212 | 0.5401 | 8.620 | 0.5336 | 0.6426 |
|  | test | 0.4541 | 0.6141 | 0.5309 | 8.653 | 0.5290 | 0.6356 |
| mlp_avg_protein | val | 0.4452 | 0.6053 | 0.5198 | 8.995 | 0.4369 | 0.6087 |
|  | test | 0.4365 | 0.6025 | 0.5153 | 8.961 | 0.4343 | 0.6041 |
| mlp_raw_protein | val | 0.4892 | 0.6135 | 0.5369 | 9.212 | 0.4955 | 0.6273 |
|  | test | 0.4862 | 0.6104 | 0.5338 | 9.197 | 0.4934 | 0.6242 |

**Best test wFmax:** `mlp_raw_protein` = 0.4862
