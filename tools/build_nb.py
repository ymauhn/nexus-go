#!/usr/bin/env python3
"""Generate the Colab notebook for the GPU-heavy PPI-only ablations (CNN1D + image)."""
import json, pathlib

cells = []
def md(src):   cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})

md("""# PPI-only — GPU-heavy ablations on Google Colab (CNN1D + image)

Runs the parts that are too slow on a local 4 GB card:
- **CNN1D** — pooling `max`/`avg`/`raw` (Ablation B, CNN1D rows). `raw` feeds a ~73k-long vector into the conv, so it's the heavy one — but fine on a Colab GPU.
- **ResNet50 & ConvNeXt-Tiny** — pooling `max`/`avg` (Ablation C).

for the **bp, cc, mf** ontologies. The MLP runs (Ablations A + B-MLP) are already done locally.

It reuses the exact `ppi_only` pipeline (batch 32, 200 epochs, patience 25, seed 1337) with **no methodology changes**, and writes per-run metrics in the **same format** as the local runs, so everything merges back into one final table.

**Flow:** pick GPU → set paths → mount Drive → stage code+data locally → verify (sanity + smoke) → run CNN1D → run image → bundle results back to Drive.""")

md("""## 1. Pick a GPU
**Runtime → Change runtime type → T4 GPU** (or better), then run the cell below.""")
code("!nvidia-smi")

md("""## 2. Configure paths  ✏️ **EDIT THESE**
Upload your `projeto` folder (with `ppi_only/`, `ppi_v4/ppi_v4/`, `configs/`) and your data folder to Google Drive, then point the variables below at them.""")
code("""# === EDIT to match your Google Drive layout ===
# Folder containing ppi_only/, ppi_v4/, configs/
PROJECT_SRC = "/content/drive/MyDrive/projeto-ppi-only/projeto"
# Folder containing ppi.csv, go.obo, {bp,cc,mf}_{train,val,test,ic}.csv
DATA_SRC    = "/content/drive/MyDrive/projeto-ppi-only/data"
# Where to write the results bundle back on Drive
RESULTS_OUT = "/content/drive/MyDrive/projeto-ppi-only/colab_results"

# Colab GPU runtimes usually have 2 vCPUs. num_workers only affects SPEED, not results
# (features are deterministic; the shuffle order is seeded in the main process).
NUM_WORKERS = 2

LOCAL = "/content/projeto"   # fast local working copy (don't edit)""")

md("""## 3. Mount Google Drive""")
code("""from google.colab import drive
drive.mount('/content/drive')""")

md("""## 4. Stage code + data onto local disk
Copies the code and data to Colab's fast local disk (Drive I/O is slow for 200-epoch training).""")
code("""import os, shutil
if os.path.exists(LOCAL):
    shutil.rmtree(LOCAL)
shutil.copytree(PROJECT_SRC, LOCAL, symlinks=False,
                ignore=shutil.ignore_patterns('runs', 'results', 'data', '.ipynb_checkpoints', '__pycache__'))
os.makedirs(f"{LOCAL}/data", exist_ok=True)
for f in os.listdir(DATA_SRC):
    if f.endswith(('.csv', '.obo')):
        shutil.copy(f"{DATA_SRC}/{f}", f"{LOCAL}/data/{f}")

assert os.path.exists(f"{LOCAL}/ppi_only/run_ablations.py"), "ppi_only/ not found under PROJECT_SRC"
assert os.path.exists(f"{LOCAL}/ppi_v4/ppi_v4/engine.py"),   "ppi_v4/ppi_v4/ not found under PROJECT_SRC"
need = {'ppi.csv', 'go.obo'} | {f"{d}_{s}.csv" for d in ('bp','cc','mf') for s in ('train','val','test','ic')}
have = set(os.listdir(f"{LOCAL}/data"))
missing = need - have
assert not missing, f"data files missing: {sorted(missing)}"
print("OK — staged at", LOCAL)
print("data:", sorted(have))""")

md("""## 5. Dependencies
Colab ships PyTorch + torchvision (CUDA). This just fills any gaps.""")
code("""import importlib.util, subprocess, sys
pip_name = {'yaml': 'pyyaml', 'sklearn': 'scikit-learn'}
need = [pip_name.get(p, p) for p in ('yaml','matplotlib','pandas','numpy','tqdm','sklearn')
        if importlib.util.find_spec(p) is None]
if need:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *need], check=True)
import torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")""")

md("""## 6. Set num_workers (results-neutral speedup)""")
code("""import re, pathlib
base = pathlib.Path(LOCAL) / "configs/ppi_only/base.yaml"
lines = base.read_text().splitlines()
lines = [re.sub(r'^num_workers:.*', f'num_workers: {NUM_WORKERS}        # Colab', ln)
         if ln.strip().startswith('num_workers:') else ln for ln in lines]
base.write_text("\\n".join(lines) + "\\n")
print("num_workers ->", NUM_WORKERS)""")

md("""## 7. Verify — sanity gate + 1-epoch smoke
If the sanity gate fails (exit code 2) or a smoke run errors, STOP and check paths/data before the full runs.""")
code("""%cd /content/projeto
!python -m ppi_only.main --sanity-only --domain bp --model resnet50""")
code("""# quick smoke: 1 epoch on a subset (one CNN1D + one image) — a couple minutes on a T4
!python -m ppi_only.main --domain bp --model cnn1d --pooling max --smoke
!python -m ppi_only.main --config configs/ppi_only/mock_image.yaml --domain bp --arch convnext_tiny --smoke""")

md("""## 8. CNN1D ablations — bp, cc, mf (9 runs)
pooling `max` / `avg` / `raw`, ProteinLoss. The three `raw` runs are the heavy ones (73k-length conv) — this is exactly what we moved off the local card.

⚠️ **Colab session limits:** free sessions disconnect after ~12 h or on idle. Bundle results (Step 10) after this cell as a checkpoint. If a session drops, re-run Steps 3–6 then continue.""")
code("""%cd /content/projeto
!python -u -m ppi_only.run_ablations --domains bp,cc,mf --only-model cnn1d""")

md("""## 9. Image ablations — bp, cc, mf (12 runs)
ResNet50 & ConvNeXt-Tiny, pooling max/avg (2 archs × 2 poolings × 3 domains). Split per-arch so you can bundle between them.""")
code("""%cd /content/projeto
!python -u -m ppi_only.run_ablations --domains bp,cc,mf --only-model resnet50""")
code("""%cd /content/projeto
!python -u -m ppi_only.run_ablations --domains bp,cc,mf --only-model convnext_tiny""")

md("""## 10. Bundle results back to Drive
Run this after CNN1D and after each image arch (checkpoint) and/or at the end.""")
code("""import os, glob, shutil, time
os.makedirs(RESULTS_OUT, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
bundle = f"/content/ppi_gpu_results_{stamp}"
for pat in ["results/*_metrics.csv", "results/*_metrics.json", "results/*_manifest.json",
            "results/ABLATIONS_summary.*", "results/_ablation_status.json",
            "runs/*_best.pt", "runs/*_losses.png", "runs/*_loss_hist.json"]:
    for f in glob.glob(f"/content/projeto/{pat}"):
        dst = os.path.join(bundle, os.path.relpath(f, "/content/projeto"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(f, dst)
zip_path = shutil.make_archive(f"{RESULTS_OUT}/ppi_gpu_results_{stamp}", "zip", bundle)
n = len(glob.glob(f"{bundle}/results/*_metrics.csv"))
print(f"Saved bundle to Drive: {zip_path}")
print(f"Per-run metric files in bundle: {n}  (expect up to 21 = 9 CNN1D + 12 image when all done)")""")

md("""## 11. Back on your local machine
1. Download `ppi_gpu_results_*.zip` from `RESULTS_OUT` on your Drive.
2. Unzip its `results/*` and `runs/*` into your local `projeto/results/` and `projeto/runs/`.
3. Tell Claude Code **"the Colab results are in"** — it will regenerate the full combined
   `results/ABLATIONS_summary.{csv,md}` + `results/RUN_REPORT.md` covering **A + B + C** (all 33 runs)
   and report all six metrics (val + test) per ablation/domain.

**Reproducibility note:** metrics are identical whether `num_workers` is 0 or 2 — features are deterministic and the DataLoader shuffle is seeded in the main process; `num_workers` changes only speed.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}

out = pathlib.Path("/home/yeonatan/Área de trabalho/projeto-ppi-only-baixar-20260724T213600Z-1-001/ppi-only/projeto/colab_gpu_experiments.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out, "with", len(cells), "cells")
