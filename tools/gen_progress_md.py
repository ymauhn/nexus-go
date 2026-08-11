#!/usr/bin/env python3
"""Build a Markdown progress report of completed vector-head runs (all 6 metrics, val+test)."""
import glob, csv, os, sys

PROJ, LOG, STAMP, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
RES = os.path.join(PROJ, "results")

METRICS = ["wfmax", "fmax", "fmax_star", "smin", "auprc", "iauprc"]
HDR     = ["wFmax", "Fmax", "Fmax*", "Smin", "AUPRC", "iAUPRC"]

# ---- progress from the run log ----
prog, cur_epoch = [], None
try:
    raw = open(LOG, errors="ignore").read().replace("\r", "\n")
    for ln in raw.splitlines():
        s = ln.strip()
        if s.startswith("###") or (s.startswith("[") and "/" in s) or "ok, " in s or "FALHOU" in s:
            prog.append(s)
    eps = [ln.strip() for ln in raw.splitlines() if ln.startswith("Epoch ")]
    cur_epoch = eps[-1] if eps else None
except FileNotFoundError:
    pass

# ---- per-run metrics ----
files = sorted(f for f in glob.glob(f"{RES}/*_metrics.csv")
               if not os.path.basename(f).startswith(("ABLATIONS", "_")))
runs = {}
for f in files:
    tag = os.path.basename(f).replace("_metrics.csv", "")
    p = tag.split("_")                     # domain, model, pooling, loss, seed
    dom, model, pool, loss = p[0], p[1], p[2], p[3]
    rows = {r["split"]: r for r in csv.DictReader(open(f))}
    runs.setdefault(dom, []).append((model, pool, loss, rows))

m_ord = {"mlp": 0, "cnn1d": 1, "resnet50": 2, "convnext": 3}
p_ord = {"max": 0, "avg": 1, "raw": 2}
def key(t):
    model, pool, loss, _ = t
    return (m_ord.get(model, 9), p_ord.get(pool, 9), 0 if loss == "bce" else 1)

def fmt(m, v):
    v = float(v)
    return f"{v:.3f}" if m == "smin" else f"{v:.4f}"

L = []
L.append("# PPI-only — Vector-head ablations (MLP + CNN1D)")
L.append("")
L.append(f"_Progress snapshot: {STAMP} · {len(files)}/21 runs with saved metrics_")
L.append("")
L.append("**Status**")
for s in prog[-9:]:
    L.append(f"- `{s}`")
if cur_epoch:
    L.append(f"- in progress: {cur_epoch}")
L.append("")
L.append("Higher is better for wFmax, Fmax, Fmax*, AUPRC, iAUPRC. **Lower is better for Smin.** "
         "`loss=protein` unless the run name says `bce`. Config: seed 1337, batch 32, 200 epochs, "
         "patience 25, `num_workers=0`. Image heads (ResNet50/ConvNeXt, Ablation C) run separately on Colab.")
L.append("")

for dom in ("bp", "cc", "mf"):
    if dom not in runs:
        continue
    L.append(f"## {dom.upper()}")
    L.append("")
    L.append("| run | split | " + " | ".join(HDR) + " |")
    L.append("|---|---|" + "---|" * len(HDR))
    best_run, best = None, -1.0
    for model, pool, loss, rows in sorted(runs[dom], key=key):
        rn = f"{model}_{pool}_{loss}"
        for sp in ("val", "test"):
            r = rows.get(sp)
            if not r:
                continue
            cells = " | ".join(fmt(m, r[m]) for m in METRICS)
            L.append(f"| {rn if sp == 'val' else ''} | {sp} | {cells} |")
        tr = rows.get("test")
        if tr and float(tr["wfmax"]) > best:
            best, best_run = float(tr["wfmax"]), rn
    L.append("")
    if best_run:
        L.append(f"**Best test wFmax:** `{best_run}` = {best:.4f}")
    L.append("")

open(OUT, "w").write("\n".join(L))
print(f"wrote {OUT} · {len(files)} runs")
