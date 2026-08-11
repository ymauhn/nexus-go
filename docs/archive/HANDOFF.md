# PPI-only ablations — SESSION HANDOFF / RESUME CONTEXT
_Last updated 2026-07-27 07:15, after the reboot. Read this first if you are a fresh Claude Code session._

## TL;DR
- **Task:** RUN (do **not** rewrite) the `ppi_only` ablation pipeline and produce
  `results/RUN_REPORT.md` + `results/ABLATIONS_summary.{csv,md}` with **all 6 metrics
  (fmax, fmax_star, wfmax, smin, auprc, iauprc) on val AND test** for the full A/B/C matrix over `bp,cc,mf`.
- **Progress:** **14 of 33 runs done locally** (all 12 MLP + `bp_cnn1d` max & avg). Saved in `results/` and `runs/`.
- **Remaining 19 runs are being done on Colab** (GPU): 9 CNN1D (incl. the slow `raw`) + 12 image. Notebook: `colab_gpu_experiments.ipynb`.
- **Colab results have NOT arrived yet** — `results/` still holds only the 14 local runs, so the final merge is not yet actionable.

## ⚠️ LOCAL GPU IS STILL DOWN — new root cause (post-reboot, 2026-07-27)
The reboot **did** clear the old `580.159` vs `580.173` mismatch, but it booted a **new kernel installed by the OS
update: `7.0.0-28-generic`**. NVIDIA now has **no kernel module at all** on this kernel — `nvidia-smi` reports
"couldn't communicate with the NVIDIA driver" and `lsmod | grep nvidia` is empty.

- `linux-modules-nvidia-580-7.0.0-28-generic` is **half-configured** (dpkg state `iF`, failed 06:53 today).
- Its postinst links prebuilt objects into `.ko` files, which needs the matching headers —
  **`linux-headers-7.0.0-28-generic` is not installed** (only 6.17.0-35 headers are). Hence
  `scripts/module.lds` / `tools/objtool` missing → link fails → only `bits/` (`.o`+`.ko.sig`), no `.ko`.
- Knock-on: `linux-modules-nvidia-580-generic-hwe-24.04` and `nvidia-driver-580` are also left unconfigured.

**Fix (needs root — sudo asks for a password, so the *user* must run it):**
```bash
sudo apt-get install -y linux-headers-7.0.0-28-generic && sudo dpkg --configure -a && sudo modprobe nvidia && nvidia-smi
```
**Fallback (no install/network):** kernel `6.17.0-35-generic` already has a fully linked
`/lib/modules/6.17.0-35-generic/kernel/nvidia-580/nvidia.ko` → reboot, GRUB → *Advanced options* → pick it.

Full evidence in `logs/agent_run.log` (entry `[env-diag 2026-07-27 07:15]`). No methodology/config changes were made.

## The goal & hard constraints (from `prompt.txt` / `instrucoes_experimentos.md`)
Run the already-built, already-tested pipeline. Reuse metric/loss/engine from `ppi_v4` via `ppi_only/compat.py`.
**Do NOT:** change the metric, the loss, or the training-only vector base; add `weight_decay`; apply a new PPI
confidence threshold; reintroduce removed blocks (`equalize`/`dilate`/`enhance`/`convexize`/ProtT5); use `raw`
pooling on image heads (vector heads only). Only fix environment/execution problems, and log them.

## Environment (verified working — NO installs needed)
- **conda env:** `deepgraphgo` — Python 3.10, torch 2.4.0+cu124, torchvision 0.19, numpy 2.2.6, matplotlib, pyyaml, pandas, scikit-learn, tqdm.
- **Activate + go to project root:**
  ```bash
  source /home/yeonatan/miniconda3/etc/profile.d/conda.sh && conda activate deepgraphgo
  cd "/home/yeonatan/Área de trabalho/projeto-ppi-only-baixar-20260724T213600Z-1-001/ppi-only/projeto"
  ```
  (or use the wrapper: `bash tools/run_in_env.sh <cmd>` — it does both, then execs `<cmd>`.)
- **Data:** `projeto/data` is a **symlink → `/home/yeonatan/Área de trabalho/data`** (ppi.csv, go.obo, `{bp,cc,mf}_{train,val,test,ic}.csv`). After reboot, verify it still resolves: `ls -L data/ppi.csv`.
- **`ppi_v4`:** at `projeto/ppi_v4/ppi_v4/` (present; `compat.py` auto-adds project root to `sys.path`).
- Run everything with cwd = the projeto/ dir so `python -m ppi_only.X` resolves.

## Config state
- `configs/ppi_only/base.yaml` → **`num_workers: 0`** (validated config; keep for local). It was briefly tried at 6 then reverted (see `logs/agent_run.log`). The Colab notebook sets it to 2. **num_workers is results-neutral** (features deterministic; shuffle seeded in main process) — only affects speed.
- No methodology changes were made.

## DONE — 14 runs (metrics saved; all 6, val+test in `results/{tag}_metrics.{csv,json}`)
```
MLP (12):  bp/cc/mf × {mlp_max_bce, mlp_max_protein, mlp_avg_protein, mlp_raw_protein}
CNN1D (2): bp_cnn1d_avg_protein, bp_cnn1d_max_protein
```
Current partial table: `PROGRESS_vector.md`. (Note: a stray `runs/bp_cnn1d_raw_*_best.pt` exists from the aborted raw run — it has **no** metrics file and will be superseded by the Colab run; ignore it.)

## REMAINING — 19 runs → run on Colab (`colab_gpu_experiments.ipynb`)
- **CNN1D (9):** `bp/cc/mf × {max, avg, raw}` — the notebook re-runs all 9 (re-doing bp max/avg is cheap and harmless; same tags/results). The 3 `raw` runs are the ones that were impractical locally.
- **Image / Ablation C (12):** `{resnet50, convnext_tiny} × {max, avg} × {bp, cc, mf}`.
- Notebook flow: upload `projeto` + `data` to Drive → set 3 path vars (Step 2) → run top-to-bottom → it bundles `ppi_gpu_results_*.zip` back to Drive.
- (`colab_image_experiments.ipynb` is the older image-only version — superseded.)

## WHY CNN1D `raw` is slow (context; do NOT change the model)
`raw` pooling feeds a **73,768-long** vector into the 2-layer 1D CNN; the conv activations stay near full length
(`(B,32,73768)`, `(B,64,36884)`) → memory-bandwidth-bound → ~**1.7 h/epoch** on the 4 GB GTX 1050 Ti (days-to-weeks
per run). `max`/`avg` use L=1024 (~72× cheaper) and finished in ~2 h each. A smaller/strided CNN would be faster
but changes the experiment — not allowed. Fix = same model on a better GPU (Colab).

## RESUME STEPS (next session)
### A) When the Colab results come back (user says "the Colab results are in")
1. Unzip the Drive bundle's `results/*` and `runs/*` into `projeto/results/` and `projeto/runs/`.
2. Build the final combined deliverables:
   ```bash
   bash tools/run_in_env.sh python tools/combine_report.py
   # (or: cd projeto && python tools/combine_report.py)
   ```
   → writes `results/ABLATIONS_summary.{csv,md}` + `results/RUN_REPORT.md` (all 6 metrics, val+test, ablations A/B/C, best wFmax(test) per ablation/domain).
3. Expect **33 runs** total (12 MLP + 9 CNN1D + 12 image). `combine_report.py` prints the count and which ablations are present.

### B) OPTIONAL — run the fast CNN1D max/avg locally **once the GPU fix above is applied** (raw stays on Colab)
Only if you want cc/mf CNN1D max/avg without waiting on Colab (Colab covers all CNN1D anyway):
```bash
for d in cc mf; do for p in max avg; do
  bash tools/run_in_env.sh python -m ppi_only.main --domain $d --model cnn1d --pooling $p
done; done
```
Each ~2 h; uses base defaults (protein loss, 200 epochs, patience 25, batch 32, seed 1337). **Do not** run `--pooling raw` locally (that's the slow one → Colab). Verify the GPU works first: `bash tools/run_in_env.sh python -c "import torch;print(torch.cuda.is_available())"` (should be `True` after reboot).

## Key files (all under projeto/)
| file | purpose |
|---|---|
| `HANDOFF.md` | this file |
| `PROGRESS_vector.md` | current 14-run metrics table |
| `colab_gpu_experiments.ipynb` | Colab notebook — CNN1D + image (the remaining 19+ runs) |
| `tools/combine_report.py` | merge all per-run metrics → final ABLATIONS_summary + RUN_REPORT |
| `tools/gen_progress_md.py` | regenerate `PROGRESS_vector.md` |
| `tools/build_nb.py` | regenerate the Colab notebook |
| `tools/run_in_env.sh` | activate `deepgraphgo` + cd projeto + exec a command |
| `logs/agent_run.log` | chronological log of every env fix/decision |
| `logs/vector_matrix.log` | stdout of the local MLP+partial-CNN1D run |

## Open decisions for the user
1. **Apply the GPU fix** (headers install, above) — required before *any* local GPU run.
2. Then: run cc/mf CNN1D max/avg locally (option B) for quicker partial results, or just let Colab do all
   CNN1D + image? (Colab covers everything either way.)
