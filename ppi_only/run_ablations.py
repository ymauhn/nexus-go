"""Orchestrate the full ablation matrix (A/B/C) and aggregate results.

Matrix (default domains: bp, cc, mf; loss=protein unless varied):
  A) MLP: BCE vs ProteinLoss          (pooling=max)
  B) MLP & CNN1D: pooling max/avg/raw  (loss=protein)
  C) ResNet50 & ConvNeXt-Tiny: max/avg (loss=protein)
Deduplicated by run tag. Ordered by increasing cost.

Usage:
  python -m ppi_only.run_ablations --domains bp,cc,mf
  python -m ppi_only.run_ablations --domains bp --smoke        # quick end-to-end dry run
"""
from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

from . import config as CFG
from .train import run_experiment

_COST_RANK = {"mlp": 0, "cnn1d": 1, "convnext_tiny": 2, "resnet50": 3}
_METRIC_KEYS = ["fmax", "fmax_star", "wfmax", "smin", "auprc", "iauprc"]


def build_matrix(domains: List[str]) -> List[Dict[str, str]]:
    specs = []
    for dom in domains:
        # A) MLP BCE vs Protein (max pooling)
        for loss in ("bce", "protein"):
            specs.append(dict(domain=dom, model="mlp", pooling="max", loss_name=loss, _ab="A"))
        # B) MLP & CNN1D pooling max/avg/raw (protein)
        for model in ("mlp", "cnn1d"):
            for pooling in ("max", "avg", "raw"):
                specs.append(dict(domain=dom, model=model, pooling=pooling, loss_name="protein", _ab="B"))
        # C) ResNet50 & ConvNeXt-Tiny pooling max/avg (protein)
        for model in ("resnet50", "convnext_tiny"):
            for pooling in ("max", "avg"):
                specs.append(dict(domain=dom, model=model, pooling=pooling, loss_name="protein", _ab="C"))

    # dedup by run tag; keep the ablation label(s)
    seen: Dict[str, Dict[str, str]] = {}
    for s in specs:
        cfg = CFG.load_config(None, domain=s["domain"], model=s["model"],
                              pooling=s["pooling"], loss_name=s["loss_name"])
        tag = CFG.make_tag(cfg)
        if tag not in seen:
            seen[tag] = {**s, "tag": tag}
        else:
            seen[tag]["_ab"] = seen[tag]["_ab"] + "+" + s["_ab"]
    ordered = sorted(seen.values(),
                     key=lambda s: (_COST_RANK.get(s["model"], 9), s["domain"], s["tag"]))
    return ordered


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="bp,cc,mf")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--ppi-csv", type=Path, default=Path("data/ppi.csv"))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only-model", default=None, help="filtra por modelo (debug)")
    ap.add_argument("--only-pooling", default=None,
                    help="filtra por pooling; aceita lista separada por vírgula (ex.: max,avg). "
                         "Útil para rodar os baratos primeiro e deixar 'raw' para depois.")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="Pula runs que já têm results/{tag}_metrics.json. SEM isto, relançar a "
                         "matriz retreina e SOBRESCREVE runs concluídos.")
    args = ap.parse_args(argv)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    matrix = build_matrix(domains)
    if args.only_model:
        matrix = [s for s in matrix if s["model"] == args.only_model]
    if args.only_pooling:
        poolings = {p.strip() for p in args.only_pooling.split(",") if p.strip()}
        matrix = [s for s in matrix if s["pooling"] in poolings]

    print(f"Matriz: {len(matrix)} runs (após dedup) — domínios {domains}")
    rows: List[Dict[str, Any]] = []
    done, failed, pulados = [], [], []

    for i, spec in enumerate(matrix, 1):
        extra = dict(smoke=True, limit_train=512, limit_val=256, max_epochs=1, patience=1) if args.smoke else {}
        cfg = CFG.load_config(None, domain=spec["domain"], model=spec["model"],
                              pooling=spec["pooling"], loss_name=spec["loss_name"], **extra)
        tag = CFG.make_tag(cfg)
        if args.skip_if_done and (args.results_dir / f"{tag}_metrics.json").exists():
            print(f"\n[{i}/{len(matrix)}] ({spec['_ab']}) {tag}  -> PULADO (metrics.json existe)")
            pulados.append(tag)
            continue
        print(f"\n[{i}/{len(matrix)}] ({spec['_ab']}) {tag}")
        try:
            res = run_experiment(cfg, args.data_dir, args.ppi_csv, args.runs_dir, args.results_dir)
            for split, m in res["metrics_by_split"].items():
                rows.append({"ablation": spec["_ab"], "domain": spec["domain"],
                             "model": spec["model"], "pooling": spec["pooling"],
                             "loss": spec["loss_name"], "split": split,
                             **{k: round(m[k], 4) for k in _METRIC_KEYS}})
            done.append(tag)
        except Exception as e:  # noqa: BLE001
            print(f"  !! FALHOU: {e}")
            traceback.print_exc()
            failed.append({"tag": tag, "error": str(e)})

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== esta invocação: {len(done)} ok, {len(pulados)} pulados, {len(failed)} falhas ===")
    for frec in failed:
        print(f"  FALHOU {frec['tag']}: {frec['error']}")

    # Os sumários vêm do DISCO, não das linhas desta invocação. Antes, rodar --only-model cnn1d
    # sobrescrevia o ABLATIONS_summary deixando só CNN1D — foi assim que o sumário do pacote ficou
    # apenas com os MLP apesar de haver 14 runs concluídos.
    from .consolidate import consolidate
    consolidate(args.results_dir)

    if failed:
        (args.results_dir / "_last_failures.json").write_text(
            json.dumps(failed, indent=2), encoding="utf-8")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
