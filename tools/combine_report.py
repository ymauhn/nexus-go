#!/usr/bin/env python3
"""Merge ALL per-run results/{tag}_metrics.csv into the final deliverables:
  results/ABLATIONS_summary.csv   (ablation,domain,model,pooling,loss,split + 6 metrics; val AND test)
  results/ABLATIONS_summary.md    (per-ablation A/B/C tables, val+test, best wFmax(test))
  results/RUN_REPORT.md

Run from the projeto/ dir:  python tools/combine_report.py
(or pass a results dir:      python tools/combine_report.py results)

Reads whatever per-run metrics are present, so it works for the partial (vector-only)
state and again after the Colab CNN1D+image results are dropped in. Reuses the exact
6 metrics the pipeline produced (fmax, fmax_star, wfmax, smin, auprc, iauprc) — it does
not recompute anything.
"""
import glob, csv, os, sys

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
KEYS = ["wfmax", "fmax", "fmax_star", "smin", "auprc", "iauprc"]   # display order
HDR  = ["wFmax", "Fmax", "Fmax*", "Smin", "AUPRC", "iAUPRC"]

def ablation(model, pool, loss):
    if model == "mlp" and pool == "max" and loss == "bce":      return "A"
    if model == "mlp" and pool == "max" and loss == "protein":  return "A+B"
    if model in ("mlp", "cnn1d") and loss == "protein":         return "B"
    if model in ("resnet50", "convnext_tiny") and loss == "protein": return "C"
    return "?"

rows = []
files = sorted(f for f in glob.glob(os.path.join(RES, "*_metrics.csv"))
               if not os.path.basename(f).startswith(("ABLATIONS", "_")))
for f in files:
    tag = os.path.basename(f).replace("_metrics.csv", "")
    p = tag.split("_")
    dom, model, pool, loss = p[0], p[1], p[2], p[3]
    ab = ablation(model, pool, loss)
    for r in csv.DictReader(open(f)):
        rows.append(dict(ablation=ab, domain=dom, model=model, pooling=pool,
                         loss=loss, split=r["split"], tag=tag,
                         **{k: float(r[k]) for k in KEYS}))

# ---- ABLATIONS_summary.csv (all 6 metrics, both splits) ----
with open(os.path.join(RES, "ABLATIONS_summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["ablation", "domain", "model", "pooling", "loss", "split"] + KEYS)
    for r in sorted(rows, key=lambda x: (x["ablation"], x["domain"], x["model"],
                                         x["pooling"], x["loss"], x["split"])):
        w.writerow([r["ablation"], r["domain"], r["model"], r["pooling"], r["loss"],
                    r["split"]] + [round(r[k], 4) for k in KEYS])

# ---- ABLATIONS_summary.md ----
md = ["# Resumo das Ablações (PPI-only) — completo", "",
      "Todas as 6 métricas em **val** e **teste**. Métrica de mérito: **wFmax(test)**. "
      "(Smin: menor é melhor; as demais: maior é melhor.)", ""]
for ab, title in [("A", "Ablação A — MLP: BCE vs ProteinLoss (pooling max)"),
                  ("B", "Ablação B — MLP/CNN1D: pooling Max/Avg/Raw (protein)"),
                  ("C", "Ablação C — ResNet50/ConvNeXt-Tiny: pooling Max/Avg (protein)")]:
    sub = [r for r in rows if ab in r["ablation"].split("+")]
    if not sub:
        continue
    md += [f"## {title}", "",
           "| domínio | modelo | pooling | loss | split | " + " | ".join(HDR) + " |",
           "|---|---|---|---|---|" + "---|" * len(HDR)]
    for r in sorted(sub, key=lambda x: (x["domain"], x["model"], x["pooling"],
                                        x["loss"], x["split"] == "test")):
        vals = " | ".join(f"{r[k]:.3f}" if k == "smin" else f"{r[k]:.4f}" for k in KEYS)
        md.append(f"| {r['domain']} | {r['model']} | {r['pooling']} | {r['loss']} | {r['split']} | {vals} |")
    tests = [r for r in sub if r["split"] == "test"]
    if tests:
        b = max(tests, key=lambda x: x["wfmax"])
        md += ["", f"**Melhor wFmax(test):** {b['wfmax']:.4f} "
                   f"({b['domain']}/{b['model']}/{b['pooling']}/{b['loss']})"]
    md.append("")
open(os.path.join(RES, "ABLATIONS_summary.md"), "w").write("\n".join(md))

# ---- RUN_REPORT.md ----
tags = sorted({r["tag"] for r in rows})
doms = sorted({r["domain"] for r in rows})
abls = sorted({r["ablation"] for r in rows})
rep = ["# RUN_REPORT — PPI-only ablations (combinado)", "",
       f"- domínios: {doms}",
       f"- runs com métricas: {len(tags)}  (esperado final: 33 = 12 MLP + 9 CNN1D + 12 imagem)",
       f"- ablações presentes: {abls}", "",
       "## Runs", *[f"- {t}" for t in tags], "",
       "## Artefatos",
       "- por run: `results/{tag}_metrics.{json,csv}`, `results/{tag}_manifest.json`",
       "- pesos/curvas: `runs/{tag}_best.pt`, `runs/{tag}_losses.png`",
       "- agregados: `results/ABLATIONS_summary.{csv,md}`, `results/RUN_REPORT.md`"]
open(os.path.join(RES, "RUN_REPORT.md"), "w").write("\n".join(rep))

print(f"combined {len(tags)} runs -> ABLATIONS_summary.{{csv,md}} + RUN_REPORT.md in '{RES}/'")
print(f"ablations present: {abls} | domains: {doms}")
