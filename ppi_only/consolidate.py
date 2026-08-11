"""Consolida TODOS os runs presentes em results/ nos sumários agregados.

Motivo de existir: `run_ablations` montava os sumários a partir apenas dos runs da invocação
atual. Rodar `--only-model cnn1d` sobrescrevia o ABLATIONS_summary deixando só CNN1D nele — foi
exatamente assim que o sumário do pacote ficou com os 12 MLP e nada mais, apesar de haver 14 runs
concluídos no disco.

Aqui a fonte da verdade é o disco: cada `results/{tag}_manifest.json` traz a `cfg` efetiva
(domain/model/pooling/loss_name) e as `metrics_by_split`. O rótulo da ablação (A/B/C) vem do
cruzamento com a matriz canônica de `run_ablations.build_matrix`.

Uso:
    python -m ppi_only.consolidate                      # results/ -> sumários
    python -m ppi_only.consolidate --results-dir results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

_METRIC_KEYS = ["fmax", "fmax_star", "wfmax", "smin", "auprc", "iauprc"]

_SECOES = [
    ("A", "Ablação A — MLP: BCE vs ProteinLoss"),
    ("B", "Ablação B — MLP/CNN1D: pooling Max/Avg/Raw"),
    ("C", "Ablação C — ResNet50/ConvNeXt-Tiny: pooling Max/Avg"),
]


def _rotulos_da_matriz() -> Dict[str, str]:
    """tag -> rótulo de ablação ('A', 'B', 'A+B', ...), da matriz canônica."""
    from .run_ablations import build_matrix
    return {s["tag"]: s["_ab"] for s in build_matrix(["bp", "cc", "mf"])}


def carregar_runs(results_dir: Path) -> List[Dict[str, Any]]:
    """Uma linha por (run, split), lida dos manifests do disco."""
    rotulos = _rotulos_da_matriz()
    rows: List[Dict[str, Any]] = []
    for p in sorted(results_dir.glob("*_manifest.json")):
        # Gates de mockup gravam com sufixo _smoke (1 época, subset) — nunca entram nas tabelas.
        if "_smoke" in p.stem:
            continue
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print(f"[AVISO] manifest ilegível, ignorado: {p.name}")
            continue
        tag = m.get("tag") or p.stem.replace("_manifest", "")
        cfg = m.get("cfg", {}) or {}
        mbs = m.get("metrics_by_split") or {}
        if not mbs:
            print(f"[AVISO] {tag}: manifest sem metrics_by_split, ignorado")
            continue
        for split, mm in mbs.items():
            if any(k not in mm for k in _METRIC_KEYS):
                print(f"[AVISO] {tag}/{split}: faltam métricas, ignorado")
                continue
            rows.append({
                "ablation": rotulos.get(tag, "?"),
                "domain": cfg.get("domain"), "model": cfg.get("model"),
                "pooling": cfg.get("pooling"), "loss": cfg.get("loss_name"),
                "split": split, "tag": tag,
                **{k: round(float(mm[k]), 4) for k in _METRIC_KEYS},
            })
    return rows


def consolidate(results_dir: Path) -> int:
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = carregar_runs(results_dir)
    tags = sorted({r["tag"] for r in rows})
    if not rows:
        print(f"[FALHA] nenhum *_manifest.json utilizável em {results_dir}")
        return 1

    with open(results_dir / "ABLATIONS_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ablation", "domain", "model", "pooling", "loss", "split", "tag"] + _METRIC_KEYS)
        for r in sorted(rows, key=lambda x: (x["ablation"], x["domain"], x["model"],
                                            x["pooling"], x["split"])):
            w.writerow([r["ablation"], r["domain"], r["model"], r["pooling"], r["loss"],
                        r["split"], r["tag"]] + [r[k] for k in _METRIC_KEYS])

    md = ["# Resumo das Ablações (PPI-only)\n",
          f"Consolidado de **{len(tags)} runs** em `{results_dir}/`. "
          "Métrica de mérito: **wFmax** (teste). As 6 métricas por run em `*_metrics.csv`.\n"]
    for ab, titulo in _SECOES:
        sub = [r for r in rows if ab in r["ablation"].split("+") and r["split"] == "test"]
        if not sub:
            md.append(f"\n## {titulo}\n\n_Nenhum run concluído._\n")
            continue
        md.append(f"\n## {titulo}\n")
        md.append("| domínio | modelo | pooling | loss | wFmax(test) | Fmax | Fmax* | Smin | AUPRC | iAUPRC |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(sub, key=lambda x: (x["domain"], x["model"], x["pooling"])):
            md.append(f"| {r['domain']} | {r['model']} | {r['pooling']} | {r['loss']} "
                      f"| {r['wfmax']:.4f} | {r['fmax']:.4f} | {r['fmax_star']:.4f} "
                      f"| {r['smin']:.4f} | {r['auprc']:.4f} | {r['iauprc']:.4f} |")
        best = max(sub, key=lambda x: x["wfmax"])
        md.append(f"\n**Melhor wFmax(test):** {best['wfmax']:.4f} "
                  f"({best['domain']}/{best['model']}/{best['pooling']}/{best['loss']})\n")
    (results_dir / "ABLATIONS_summary.md").write_text("\n".join(md), encoding="utf-8")

    # Cobertura: o que da matriz canônica ainda não existe no disco.
    esperados = set(_rotulos_da_matriz())
    faltando = sorted(esperados - set(tags))
    rep = ["# RUN_REPORT — PPI-only ablations\n",
           f"- runs no disco: **{len(tags)}** de {len(esperados)} da matriz canônica",
           f"- pendentes: **{len(faltando)}**"]
    if faltando:
        rep.append("\n## Pendentes")
        for t in faltando:
            rep.append(f"- `{t}`")
    rep.append("\n## Concluídos")
    for t in tags:
        rep.append(f"- `{t}`")
    rep.append("\n## Artefatos por run")
    rep.append("- pesos/curvas: `runs/{tag}_best.pt`, `runs/{tag}_losses.png`, `runs/{tag}_loss_hist.json`")
    rep.append("- métricas: `results/{tag}_metrics.{json,csv}`, `results/{tag}_manifest.json`")
    rep.append("- agregados: `results/ABLATIONS_summary.{csv,md}`")
    (results_dir / "RUN_REPORT.md").write_text("\n".join(rep), encoding="utf-8")

    (results_dir / "_ablation_status.json").write_text(
        json.dumps({"done": tags, "pending": faltando, "failed": []}, indent=2), encoding="utf-8")

    print(f"[OK] {len(tags)} runs consolidados ({len(rows)} linhas run x split) -> "
          f"{results_dir}/ABLATIONS_summary.{{csv,md}}")
    if faltando:
        print(f"[i] pendentes na matriz canônica ({len(faltando)}): {', '.join(faltando)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    return consolidate(args.results_dir)


if __name__ == "__main__":
    raise SystemExit(main())
